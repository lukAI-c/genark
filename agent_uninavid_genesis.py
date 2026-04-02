import numpy as np
import os
import re
import torch
import cv2
import imageio
import random

# Removed habitat imports
# from habitat.core.agent import Agent
# from habitat.utils.visualizations import maps

from uninavid.mm_utils import get_model_name_from_path
from uninavid.model.builder import load_pretrained_model
from uninavid.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from uninavid.conversation import conv_templates, SeparatorStyle
from uninavid.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria
from uninavid.constants import NAVIGATION_IDENTIFIER
import uninavid
  

# Removed Agent inheritance
class UniNaVid_Agent:
    def __init__(self, model_path, result_path, exp_save, save_debug_images=False):
        
        print("Initialize UniNaVid (Genesis Version)")

        self.result_path = result_path
        self.require_map = True if "video" in exp_save else False
        self.require_data = True if "data" in exp_save else False
        self.save_debug_images = save_debug_images

        self.conv_mode = "vicuna_v1"
        
        if self.require_map or self.require_data:
            os.makedirs(self.result_path, exist_ok=True)
        
        if self.require_data:
            os.makedirs(os.path.join(self.result_path, "log"), exist_ok=True)
        
        if self.require_data:
            os.makedirs(os.path.join(self.result_path, "video"), exist_ok=True)


        self.model_name = get_model_name_from_path(model_path)
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(model_path, None, get_model_name_from_path(model_path))

        # --- DEBUG: Tokenizer & Model Check ---
        vocab_size = len(self.tokenizer)
        embedding_size = self.model.get_input_embeddings().weight.shape[0]
        # print(f"[Agent Debug] Tokenizer Vocab Size: {vocab_size}")
        # print(f"[Agent Debug] Model Embedding Size: {embedding_size}")
        
        if vocab_size != embedding_size:
            print(f"[Agent Warning] Vocab size mismatch! Model has {embedding_size} embeddings, but tokenizer has {vocab_size} tokens.")
            if embedding_size > vocab_size:
                print(f"[Agent Suggestion] There are {embedding_size - vocab_size} missing tokens in the tokenizer. Try adding special tokens.")
        
        self.model.eval() # Ensure model is in eval mode
        # print(f"[Agent Debug] Model set to eval() mode.")
        # --------------------------------------

        # --- DEBUG: Check Vision Tower ---
        vision_tower = self.model.get_vision_tower()
        # print(f"[Agent Debug] Vision Tower: {vision_tower}")
        if vision_tower is None:
            print("[Critical Error] Vision Tower is None! The model is running as pure LLM.")
        else:
            pass
            # print(f"[Agent Debug] Vision Tower Config: {vision_tower.config if hasattr(vision_tower, 'config') else 'No Config'}")
        # ---------------------------------
        
        
        print(f"Initialization Complete")
        # print(f"[Agent Debug] uninavid package location: {os.path.dirname(uninavid.__file__)}")

        
        self.promt_template = "Imagine you are a robot programmed for navigation tasks. You have been given a video of historical observations and an image of the current observation <image>. Your assigned task is: '{}'. Analyze this series of images to determine your next four actions. The predicted action should be one of the following: forward, left, right, or stop."
        # Note: If the model is not following instructions, we might need to adjust this template or special tokens.
        # Check if model expects <image> token? Yes, it's in the template.

        # self.history_rgb_tensor = None
        
        self.rgb_list = []
        self.topdown_map_list = []
        
        self.full_history_log = [] # For debugging only
        
        self.count_id = 0
        self.reset()


    def process_images(self, rgb_list):
        
        # start_img_index = 0
        
        # if self.history_rgb_tensor is not None:
            # start_img_index = self.history_rgb_tensor.shape[0]
        
        # Convert list of images to numpy array
        # Note: If images are different sizes, np.asarray might create an object array, which could fail later
        # We ensure they are stacked properly.
        if len(rgb_list) == 0:
             return None
             
        try:
            batch_image = np.stack(rgb_list) # Use stack instead of asarray for list of arrays
        except ValueError as e:
            print(f"[Error] Failed to stack images: {e}")
            raise e
        
        # --- DEBUG: Check Batch Image Stats ---
        # print(f"[Agent Debug] Batch Image Stats (before preprocess) - Mean: {np.mean(batch_image):.2f}, Max: {np.max(batch_image)}, Min: {np.min(batch_image)}, Dtype: {batch_image.dtype}")
        # --------------------------------------

        self.model.get_model().new_frames = len(rgb_list)
        
        # Check if batch_image is valid
        if batch_image is None or batch_image.size == 0:
             print("[Error] Batch image is empty!")
             return None
             
        # video = self.image_processor.preprocess(batch_image, return_tensors='pt')['pixel_values'].half().cuda()
        video = self.image_processor.preprocess(batch_image, return_tensors='pt')['pixel_values'].to(dtype=self.model.dtype, device='cuda')

        # --- DEBUG: Check Video Tensor Stats ---
        # print(f"[Agent Debug] Video Tensor Stats (after preprocess) - Mean: {video.mean().item():.2f}, Max: {video.max().item():.2f}, Min: {video.min().item():.2f}, Shape: {video.shape}")
        # ---------------------------------------

        return [video]


    def predict_inference(self, prompt, step_idx=None):
        
        question = prompt.replace(DEFAULT_IMAGE_TOKEN, '').replace('\n', '')
        qs = prompt

        VIDEO_START_SPECIAL_TOKEN = "<video_special>"
        VIDEO_END_SPECIAL_TOKEN = "</video_special>"
        IMAGE_START_TOKEN = "<image_special>"
        IMAGE_END_TOKEN = "</image_special>"
        NAVIGATION_SPECIAL_TOKEN = "[Navigation]"
        IAMGE_SEPARATOR = "<image_sep>"
        
        # Safe token extraction
        def get_special_token(token_str):
            # Check if token exists in vocab
            if token_str not in self.tokenizer.get_vocab() and token_str not in self.tokenizer.get_added_vocab():
                 print(f"[Agent Warning] Special token '{token_str}' not found in tokenizer vocab! It will be split.")

            # Try direct ID lookup first (safest for special tokens)
            id_val = self.tokenizer.convert_tokens_to_ids(token_str)
            
            # If it returns UNK (and it's not actually UNK token), then it failed
            if id_val != self.tokenizer.unk_token_id or token_str == self.tokenizer.unk_token:
                 # print(f"[Agent Debug] Token '{token_str}' -> Direct ID: {id_val}")
                 return torch.tensor([id_val], device='cuda')
            
            # Fallback to encoding (original logic, but careful about spaces)
            ids = self.tokenizer(token_str, return_tensors="pt", add_special_tokens=False).input_ids[0]
            # print(f"[Agent Debug] Token '{token_str}' -> Encoded IDs: {ids.tolist()}")
            
            # Filter out leading space (Llama underscore: 29871) if present and we have multiple tokens
            if len(ids) > 1 and ids[0] == 29871:
                 # print(f"[Agent Debug] Removing leading space from '{token_str}'")
                 ids = ids[1:]
                 
            if len(ids) == 0:
                ids = self.tokenizer(token_str, return_tensors="pt").input_ids[0]
                if ids[0] == self.tokenizer.bos_token_id:
                    ids = ids[1:]
            return ids.cuda()

        image_start_special_token = get_special_token(IMAGE_START_TOKEN)
        image_end_special_token = get_special_token(IMAGE_END_TOKEN)
        video_start_special_token = get_special_token(VIDEO_START_SPECIAL_TOKEN)
        video_end_special_token = get_special_token(VIDEO_END_SPECIAL_TOKEN)
        navigation_special_token = get_special_token(NAVIGATION_SPECIAL_TOKEN)
        image_seperator = get_special_token(IAMGE_SEPARATOR)

        if self.model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs.replace('<image>', '')
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs.replace('<image>', '')

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        token_prompt = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').cuda()
        indices_to_replace = torch.where(token_prompt == -200)[0]
        new_list = []
        while indices_to_replace.numel() > 0:
            idx = indices_to_replace[0]
            new_list.append(token_prompt[:idx])
            new_list.append(video_start_special_token)
            new_list.append(image_seperator)
            new_list.append(token_prompt[idx:idx + 1])
            new_list.append(video_end_special_token)
            new_list.append(image_start_special_token)
            new_list.append(image_end_special_token)
            new_list.append(navigation_special_token)
            token_prompt = token_prompt[idx + 1:]
            indices_to_replace = torch.where(token_prompt == -200)[0]
        if token_prompt.numel() > 0:
            new_list.append(token_prompt)
        input_ids = torch.cat(new_list, dim=0).unsqueeze(0)

        # --- DEBUG: Check Prompt Tokens ---
        # print(f"[Agent Debug] Input IDs shape: {input_ids.shape}")
        # Replace -200 (IMAGE_TOKEN_INDEX) with 0 (unk) just for decoding/printing
        debug_ids = input_ids[0].clone()
        debug_ids[debug_ids < 0] = 0 
        decoded_prompt = self.tokenizer.decode(debug_ids, skip_special_tokens=False)
        # Truncate for readability if too long
        # print(f"[Agent Debug] Decoded Prompt (First 200 chars): {decoded_prompt[:200]}...")
        # print(f"[Agent Debug] Decoded Prompt (Last 200 chars): {decoded_prompt[-200:]}")
        # ----------------------------------

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)

        # imgs = self.process_images(self.rgb_list)
        # self.rgb_list = []
        
        # NOTE: self.rgb_list contains all frames since reset? 
        # Or just new frames?
        # Looking at act(): self.rgb_list.append(rgb)
        # So it accumulates.
        # But process_images() sets self.model.get_model().new_frames = len(rgb_list)
        # If we clear self.rgb_list here, next time we only process new frames?
        # Original code cleared it: self.rgb_list = []
        
        # Let's check what process_images returns. It returns [video] tensor.
        
        # Debugging step: Check if self.rgb_list contains None or invalid items
        for i, img in enumerate(self.rgb_list):
            if img is None:
                print(f"[Error] self.rgb_list[{i}] is None!")
                return "stop"
            if not isinstance(img, (np.ndarray, torch.Tensor)):
                 print(f"[Error] self.rgb_list[{i}] is {type(img)} not array!")
                 return "stop"
        
        # --- DEBUG: Save Historical & Current Observations ---
        if self.save_debug_images and step_idx is not None:
             debug_dir = f"debug_images/step_{step_idx}"
             os.makedirs(debug_dir, exist_ok=True)
             
             # The last image is Current Observation
             # The previous images are Historical Observations (for this inference step)
             
             # Use full_history_log for complete visualization
             history_source = self.full_history_log
             num_images = len(history_source)
             
             for i, img in enumerate(history_source):
                 # Convert RGB to BGR for cv2
                 if isinstance(img, np.ndarray):
                      img_bgr = img[..., ::-1]
                 elif isinstance(img, torch.Tensor):
                      img_bgr = img.cpu().numpy()[..., ::-1]
                 else:
                      continue
                      
                 if i == num_images - 1:
                      # Current
                      cv2.imwrite(os.path.join(debug_dir, "current_obs.png"), img_bgr)
                 else:
                      # History
                      cv2.imwrite(os.path.join(debug_dir, f"history_obs_{i}.png"), img_bgr)
             
             # print(f"[Agent Debug] Saved {num_images} input images to {debug_dir}")
        # ---------------------------------------------------

        imgs = self.process_images(self.rgb_list)
        self.rgb_list = [] # Clear the buffer after processing
        
        # imgs[0] is the video tensor from process_images
        # It should have shape (B, C, T, H, W) or similar
        # If the model expects a list of tensors, we should be fine.
        # But if process_images returns [video_tensor], then imgs[0] IS the tensor.
        
        # Check for NoneType inside the list
        if imgs is None or len(imgs) == 0 or imgs[0] is None:
             print("[Error] Processed images are None or empty. Skipping generation.")
             return "stop"
             
        cur_prompt = question
        
        # --- DEBUG PRINT: Input Prompt & Identifier Check ---
        # print(f"\n[Agent Input] Step {self.count_id}, Prompt: {cur_prompt}")
        
        if NAVIGATION_IDENTIFIER in cur_prompt:
             pass
             # print("[Agent Debug] Navigation Identifier FOUND in prompt.")
        else:
             # print("[Agent Debug] Navigation Identifier NOT FOUND in prompt!")
             # print(f"[Agent Debug] Expected: '{NAVIGATION_IDENTIFIER}'")
             # print(f"[Agent Debug] Actual (first 100): '{cur_prompt[:100]}'")
             pass
             
        # ---------------------------------
        
        # Try different decoding settings
        with torch.inference_mode():
            # print(f"[Agent Debug] cur_prompt for update_prompt: {cur_prompt[:100]}... (len={len(cur_prompt)})")
            self.model.update_prompt([[cur_prompt]])
            
            # --- DEBUG: Verify self.model.prompts ---
            # if hasattr(self.model, 'prompts'):
            #     print(f"[Agent Debug] self.model.prompts set to list of len {len(self.model.prompts)}")
            # else:
            #     print(f"[Agent Debug] self.model does NOT have prompts attribute!")
            # 
            # model_base = self.model.get_model()
            # if hasattr(model_base, 'prompts'):
            #     print(f"[Agent Debug] self.model.get_model().prompts set to list of len {len(model_base.prompts)}")
            # ----------------------------------------

            output_ids = self.model.generate(
                input_ids,
                images=imgs,
                do_sample=True, # Restore to True as per original code
                num_beams=1,
                temperature=0.2, # Restore temperature
                max_new_tokens=1024,
                use_cache=True,
                stopping_criteria=[stopping_criteria])

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = self.tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()
        
        # --- DEBUG PRINT: Model Output ---
        # print(f"[Agent Output] Raw Prediction: {outputs}")
        # ---------------------------------

        return outputs



    def extract_result(self, output):
        # id: 0-stop, 1 move forward, 2 turn left, 3 turn right

        if "stop" in output:
            return 0, None
        elif "forward" in output:
            match = re.search(r'-?\d+', output)
            if match is None:
                return None, None
            match = match.group()
            return 1, float(match)
        elif "left" in output:
            match = re.search(r'-?\d+', output)
            if match is None:
                return None, None
            match = match.group()
            return 2, float(match)
        elif "right" in output:
            match = re.search(r'-?\d+', output)
            if match is None:
                return None, None
            match = match.group()
            return 3, float(match)

        return None, None



    def addtext(self, image, instuction, navigation):
        h, w = image.shape[:2]
        new_height = h + 150
        new_image = np.zeros((new_height, w, 3), np.uint8)
        new_image.fill(255)  
        new_image[:h, :w] = image

        font = cv2.FONT_HERSHEY_SIMPLEX
        textsize = cv2.getTextSize(instuction, font, 0.5, 2)[0]
        textY = h + (50 + textsize[1]) // 2

        y_line = textY + 0 * textsize[1]



        words = instuction.split(' ')
        max_width = new_image.shape[1]
        x = 10
        line = ""

        for word in words:

            test_line = line + ' ' + word if line else word
            test_line_size, _ = cv2.getTextSize(test_line, font, 0.5, 2)

            if test_line_size[0] > image.shape[1] - x:
                cv2.putText(new_image, line, (x, y_line ), font, 0.5, (0, 0, 0), 2)
                line = word
                y_line += textsize[1]+5
            else:
                line = test_line


        if line:

            cv2.putText(new_image, line, (x, y_line), font, 0.5, (0, 0, 0), 2)


        y_line = y_line + 1 * textsize[1] + 10
        new_image = cv2.putText(new_image, navigation, (x, y_line), font, 0.5, (0, 0, 0), 2)

        return new_image


    def reset(self):
                
        if self.require_map:
            # Removed map saving logic as it depends on habitat maps
            # if len(self.topdown_map_list)!=0:
            #     output_video_path = os.path.join(self.result_path, "video","{}.gif".format(self.episode_id))
            #     imageio.mimsave(output_video_path, self.topdown_map_list)
            pass

        self.history_rgb_tensor = None
        self.transformation_list = []
        self.rgb_list = []
        self.full_history_log = [] # Reset debug log
        self.topdown_map_list = []
        self.last_action = None
        self.count_id += 1
        self.count_stop = 0
        self.pending_action_list = []

        self.model.config.run_type = "eval"
        
        # Restore original cache initialization logic
        # This is critical if the model relies on these structures being present (even if empty)
        if hasattr(self.model.get_model(), "initialize_online_inference_nav_feat_cache"):
             # print("[Agent Debug] Initializing online inference nav feat cache")
             self.model.get_model().initialize_online_inference_nav_feat_cache()
        else:
             print("[Agent Warning] Model does not have initialize_online_inference_nav_feat_cache method!")
             
        self.model.get_model().new_frames = 0
        
        self.first_forward = False
        


    def act(self, observations, info, episode_id, step_idx=None):
        try:
            # --- DEBUG: Runtime Vision Tower Check ---
            vt = self.model.get_vision_tower()
            if vt is None:
                 print(f"[Critical Error] Vision Tower is None in act()! Model Name: {self.model_name}")
            # -----------------------------------------

            self.episode_id = episode_id
            rgb = observations["rgb"]
            
            # --- DEBUG: RGB Input ---
            if rgb is None:
                 print("[Agent Debug] Received RGB is None!")
            else:
                 pass
                 # print(f"[Agent Debug] Received RGB shape: {rgb.shape}, dtype: {rgb.dtype}")
                 # print(f"[Agent Debug] RGB Stats - Min: {rgb.min()}, Max: {rgb.max()}, Mean: {rgb.mean():.2f}")
            # ------------------------
            
            self.rgb_list.append(rgb)
            self.full_history_log.append(rgb)

            output_im = None
            if self.require_map:
                # maps logic disabled
                # top_down_map = maps.colorize_draw_agent_and_fit_to_height(info["top_down_map_vlnce"], rgb.shape[0])
                # output_im = np.concatenate((rgb, top_down_map), axis=1)
                output_im = rgb # Fallback

            if len(self.pending_action_list) != 0:
                temp_action = self.pending_action_list.pop(0)
            
                if self.require_map and output_im is not None:
                    img = self.addtext(output_im, observations["instruction"]["text"], "Pending action: {}".format(temp_action))
                    self.topdown_map_list.append(img)
            
                return {"action": temp_action}
            
            # Debug before predict
            # print(f"[Agent Debug] Calling predict_inference with prompt len: {len(self.promt_template.format(observations['instruction']['text']))}")


            navigation_qs = self.promt_template.format(observations["instruction"]["text"])
            
      
            navigation = self.predict_inference(navigation_qs, step_idx=step_idx)
            
            if self.require_map and output_im is not None:
                img = self.addtext(output_im, observations["instruction"]["text"], navigation)
                self.topdown_map_list.append(img)


            action_list = navigation.split(" ")

            for action in action_list: 
                if action == "stop":
                    self.pending_action_list.append(0)
                elif action == "forward":
                    self.pending_action_list.append(1)
                elif action == "left":
                    self.pending_action_list.append(2)
                elif action == "right":
                    self.pending_action_list.append(3)
                # else:
                #     # Tolerant to other outputs or noise
                #     pass
                                        
                if len(self.pending_action_list) == 2: # to accelrate the inference
                    break

            if len(self.pending_action_list) == 0:
                # Fallback if no valid action parsed
                print(f"Warning: No valid action parsed from: '{navigation}'. Defaulting to STOP.")
                return {"action": 0}

            next_action = self.pending_action_list.pop(0)
            
            # --- DEBUG PRINT: Parsed Action ---
            action_str = {0: "STOP", 1: "FORWARD", 2: "LEFT", 3: "RIGHT"}.get(next_action, "UNKNOWN")
            # print(f"[Agent Action] Executing: {action_str} (ID: {next_action})\n")
            # ----------------------------------
            
            return {"action": next_action}
        
        except Exception as e:
              import traceback
              print(f"[Critical Error] Agent act failed: {e}")
              traceback.print_exc()
              raise e
