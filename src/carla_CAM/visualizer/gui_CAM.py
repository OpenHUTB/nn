"""
Visualisation API to produce CAM overlayed images

Offers a GUI through Pygame to facilitate the evaluations

@author: Pablo Roca - github.com/RocaPiedra
"""

import sys
import os
import random
import numpy as np
from PIL import Image

import pygame
from pygame.locals import *
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
sys.path.append('../visualizer')
from roc_functions import draw_text

import torch
from torchvision import transforms
from torchvision.models import resnet18, resnet34, resnet50, alexnet, vgg11, vgg19
import parameters

import roc_functions
import time
import gc

class gui_CAM:
    def __init__(self, display, use_cuda = True, display_manager = None):
        self.model = resnet34(pretrained=True)
        self.model_gradient_compatible = True
        self.display_manager = display_manager
        self.display = display
        try:
            roc_functions.blip_logo(self.display, parameters.gui_cam_logo)
        except Exception as e:
            print(f'Logo blip failed with exception:\n{e}')
        self.use_cuda = use_cuda
        self.font = pygame.font.SysFont(None, 24)
        self.target_layers = self.select_target_layer()
        self.cam_name = None
        self.method_name = None
        self.model_name = 'ResNet'
        self.NONGRAD_BUTTON_COLOR = parameters.BUTTON_COLOR
        self.GRAD_BUTTON_COLOR = parameters.BUTTON_COLOR
        self.MODEL_BUTTON_COLOR = parameters.BUTTON_COLOR
        self.click = False
        self.class_list = roc_functions.get_imagenet_dictionary(url=None) 
        self.cam = None
        self.classification_output = ''  
        self.filtered_cam = True #Determines if the target classes are filtered for the desired ones
        if torch.cuda.is_available() and self.use_cuda:
            self.model.to('cuda')
            print("System is cuda ready")
            
        self.surface = None
        self.main_location = None
        self.display_width, self.display_height = pygame.display.get_surface().get_size()
        #location where second cam is plotted in the display
        
        self.compare_location = [int(2*self.display_width/3), int(self.display_height/2)]

        
    def select_cam(self, second_method = False):            
        method_selection = True
        x = 100
        y = 100
        dx = 0
        dy = 100
        w, h = pygame.display.get_surface().get_size()
        button_width = 300
        button_height = 40
        num_options = 7
        positions = []
        if not self.model_gradient_compatible:
            self.GRAD_BUTTON_COLOR = parameters.RED_BUTTON_COLOR
        else:
            self.GRAD_BUTTON_COLOR = parameters.BUTTON_COLOR
                
        for pos in range(num_options):
            
            positions.append([x+pos*dx, y+pos*dy])
        
        draw_text('CAM Technique Menu', self.font, (255, 255, 255), self.display, 20, 20)
        
        # To delimit the size of the button, in the future use value related to window res
        
        grad_button = pygame.Rect(positions[0][0], positions[0][1], button_width, button_height)
        score_button = pygame.Rect(positions[4][0], positions[4][1], button_width, button_height)
        xgradcam_button = pygame.Rect(positions[2][0], positions[2][1], button_width, button_height)
        ablation_button = pygame.Rect(positions[5][0], positions[5][1], button_width, button_height)
        eigen_button = pygame.Rect(positions[6][0], positions[6][1], button_width, button_height)
        fullgrad_button = pygame.Rect(positions[3][0], positions[3][1], button_width, button_height)
        gradcampp_button = pygame.Rect(positions[1][0], positions[1][1], button_width, button_height)

        pygame.draw.rect(self.display, self.GRAD_BUTTON_COLOR, grad_button)
        draw_text('GradCAM', self.font, (255, 255, 255), self.display, positions[0][0]+10, positions[0][1]+button_height-20)
        pygame.draw.rect(self.display, self.NONGRAD_BUTTON_COLOR, score_button)
        draw_text('ScoreCAM', self.font, (255, 255, 255), self.display, positions[4][0]+10, positions[4][1]+button_height-20)
        pygame.draw.rect(self.display, self.GRAD_BUTTON_COLOR, xgradcam_button)
        draw_text('XGradCAM', self.font, (255, 255, 255), self.display, positions[2][0]+10, positions[2][1]+button_height-20)
        pygame.draw.rect(self.display, self.NONGRAD_BUTTON_COLOR, ablation_button)
        draw_text('AblationCAM', self.font, (255, 255, 255), self.display, positions[5][0]+10, positions[5][1]+button_height-20)
        pygame.draw.rect(self.display, self.NONGRAD_BUTTON_COLOR, eigen_button)
        draw_text('EigenCAM', self.font, (255, 255, 255), self.display, positions[6][0]+10, positions[6][1]+button_height-20)
        pygame.draw.rect(self.display, self.GRAD_BUTTON_COLOR, fullgrad_button)
        draw_text('FullGrad', self.font, (255, 255, 255), self.display, positions[3][0]+10, positions[3][1]+button_height-20)
        pygame.draw.rect(self.display, self.GRAD_BUTTON_COLOR, gradcampp_button)
        draw_text('GradCAM++', self.font, (255, 255, 255), self.display, positions[1][0]+10, positions[1][1]+button_height-20)
        
        pygame.display.update()
        
        while method_selection:
            
            mx, my = pygame.mouse.get_pos()
            if grad_button.collidepoint((mx, my)):
                if self.click and self.model_gradient_compatible:
                    method_selection = False
                    method_name = 'GradCAM'
                elif self.click and not self.model_gradient_compatible:
                    print(f'The method is gradient based and not compatible with the current model {self.model_name}')
            if score_button.collidepoint((mx, my)):
                if self.click:
                    method_selection = False
                    method_name = 'ScoreCAM'
                    
            if ablation_button.collidepoint((mx, my)):
                if self.click:
                    method_selection = False
                    method_name = 'AblationCAM'
                    
            if xgradcam_button.collidepoint((mx, my)):
                if self.click and self.model_gradient_compatible:
                    method_selection = False
                    method_name = 'XGradCAM'
                elif self.click and not self.model_gradient_compatible:
                    print(f'The method is gradient based and not compatible with the current model {self.model_name}')
                    
            if eigen_button.collidepoint((mx, my)):
                if self.click:
                    method_selection = False
                    method_name = 'EigenCAM'
                    
            if fullgrad_button.collidepoint((mx, my)):
                if self.click and self.model_gradient_compatible:
                    method_selection = False
                    method_name = 'FullGrad'
                elif self.click and not self.model_gradient_compatible:
                    print(f'The method is gradient based and not compatible with the current model {self.model_name}')
                    
            if gradcampp_button.collidepoint((mx, my)):
                if self.click and self.model_gradient_compatible:
                    method_selection = False
                    method_name = 'GradCAM++'
                elif self.click and not self.model_gradient_compatible:
                    print(f'The method is gradient based and not compatible with the current model {self.model_name}')
                    
            self.click = False
            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == KEYDOWN:
                    if event.key == K_ESCAPE:
                        pygame.quit()
                        sys.exit()
                if event.type == MOUSEBUTTONDOWN:
                    if event.button == 1:
                        self.click = True
        if second_method is False:
            self.method_name = method_name
        return method_name    
    
    
    def load_cam(self, cuda_error = False, method_name=None):
        
        if not method_name:
            method_name = self.method_name
            
        if method_name == 'ScoreCAM':
            if not cuda_error:
                cam_method = ScoreCAM(model=self.model, target_layers=self.target_layers, use_cuda=True)
            else:
                print("error thrown, using CPU")
                cam_method = ScoreCAM(model=self.model, target_layers=self.target_layers, use_cuda=False)
            
        if method_name == 'AblationCAM':
            if not cuda_error:
                cam_method = AblationCAM(model=self.model, target_layers=self.target_layers, use_cuda=True)
            else:
                print("error thrown, using CPU")
                cam_method = AblationCAM(model=self.model, target_layers=self.target_layers, use_cuda=False)
            
        if method_name == 'XGradCAM':
            if not cuda_error:
                cam_method = XGradCAM(model=self.model, target_layers=self.target_layers, use_cuda=True)
            else:
                print("error thrown, using CPU")
                cam_method = XGradCAM(model=self.model, target_layers=self.target_layers, use_cuda=False)
            
        if method_name == 'EigenCAM':
            if not cuda_error:
                cam_method = EigenCAM(model=self.model, target_layers=self.target_layers, use_cuda=True)
            else:
                print("error thrown, using CPU")
                cam_method = EigenCAM(model=self.model, target_layers=self.target_layers, use_cuda=False)
            
        if method_name == 'FullGrad':
            if not cuda_error:
                cam_method = FullGrad(model=self.model, target_layers=self.target_layers, use_cuda=True)
            else:
                print("error thrown, using CPU")
                cam_method = FullGrad(model=self.model, target_layers=self.target_layers, use_cuda=False)
            
        if method_name == 'GradCAM++':
            if not cuda_error:
                cam_method = GradCAMPlusPlus(model=self.model, target_layers=self.target_layers, use_cuda=True)
            else:
                print("error thrown, using CPU")
                cam_method = GradCAMPlusPlus(model=self.model, target_layers=self.target_layers, use_cuda=False)
            
        if method_name == 'GradCAM':
            if not cuda_error:
                cam_method = GradCAM(model=self.model, target_layers=self.target_layers, use_cuda=True)
            else:
                print("error thrown, using CPU")
                cam_method = GradCAM(model=self.model, target_layers=self.target_layers, use_cuda=False)
            
        if cam_method:    
            return cam_method
        else:
            self.select_cam()
    
    
    def select_target_layer(self):
        # to implement for multiple models
        if self.model.__class__.__name__ == 'ResNet':
            self.target_layers = [self.model.layer4[-1]]
            print(f'Target Layer for {self.model.__class__.__name__} is:')
            
        elif self.model.__class__.__name__ == 'Alexnet':
            self.target_layers = [11]
            print(f'Target Layer for {self.model.__class__.__name__} is:')
            
        elif self.model.__class__.__name__ == 'VGG':
            self.target_layers = [self.model.features[-1]]
            print(f'Target Layer for {self.model.__class__.__name__} is:')
            
        elif self.model.__class__.__name__ == 'AutoShape':
            self.target_layers = [self.model.model.model.model[-2]]
            print(f'Target Layer for YOLOv5 is:')
            
        # print(self.target_layers)    
        return self.target_layers
    
    
    def select_model(self):
        model_selection = True
        x = 100
        y = 100
        dx = 0
        dy = 100
        num_options = 4 
        positions = []
        for pos in range(num_options):
            positions.append([x+pos*dx, y+pos*dy])
        
        draw_text('Model Menu', self.font, (255, 255, 255), self.display, 20, 20)
        
        # To delimit the size of the button, in the future use value related to window res
        w, h = pygame.display.get_surface().get_size()
        button_width = 300
        button_height = 40
        
        resnet_button = pygame.Rect(positions[0][0], positions[0][1], button_width, button_height)
        alexnet_button = pygame.Rect(positions[1][0], positions[1][1], button_width, button_height)
        third_button = pygame.Rect(positions[2][0], positions[2][1], button_width, button_height)
        fourth_button = pygame.Rect(positions[3][0], positions[3][1], button_width, button_height)

        pygame.draw.rect(self.display, self.MODEL_BUTTON_COLOR,  resnet_button)
        draw_text('ResNet', self.font, (255, 255, 255), self.display, positions[0][0]+10, positions[0][1]+button_height-20)
        pygame.draw.rect(self.display, self.MODEL_BUTTON_COLOR, alexnet_button)
        draw_text('Alexnet', self.font, (255, 255, 255), self.display, positions[1][0]+10, positions[1][1]+button_height-20)
        pygame.draw.rect(self.display, self.MODEL_BUTTON_COLOR, third_button)
        draw_text('VGG', self.font, (255, 255, 255), self.display, positions[2][0]+10, positions[2][1]+button_height-20)
        pygame.draw.rect(self.display, parameters.RED_BUTTON_COLOR, fourth_button)
        draw_text('YOLOv5', self.font, (255, 255, 255), self.display, positions[3][0]+10, positions[3][1]+button_height-20)
        
        pygame.display.update()
        
        while model_selection:
            
            mx, my = pygame.mouse.get_pos()    
            if resnet_button.collidepoint((mx, my)):
                if self.click:
                    if self.model.__class__.__name__ != 'ResNet':
                        self.clear_memory()
                        self.model = resnet34(pretrained=True)
                        model_selection = False
                        model_name = 'ResNet'
                        self.model_gradient_compatible = True
                        
                    else:
                        print(f'Model selected -> {model_name} was already loaded')
            
            if alexnet_button.collidepoint((mx, my)):
                if self.click:
                    if self.model.__class__.__name__ != 'Alexnet':
                        self.clear_memory()
                        self.model = alexnet(pretrained=True) 
                        model_selection = False
                        model_name = 'Alexnet'
                        self.model_gradient_compatible = True
                        
                    else:
                        print(f'Model selected -> {model_name} was already loaded')
            
            if third_button.collidepoint((mx, my)):
                if self.click:
                    if self.model.__class__.__name__ != 'VGG':
                        self.clear_memory()
                        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg11', pretrained=True)
                        model_selection = False
                        model_name = 'VGG'
                        self.model_gradient_compatible = True
                        
                    else:
                        print(f'Model selected -> {model_name} was already loaded')
                    
            if fourth_button.collidepoint((mx, my)):
                if self.click:
                    # not sure why, yolov5 returns as name AutoShape.
                    if self.model.__class__.__name__ != 'AutoShape':
                        self.clear_memory()
                        self.model =  torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)        
                        model_selection = False
                        model_name = 'YOLOv5'
                        self.model_gradient_compatible = False
                        
                    else:
                        print(f'Model selected -> {model_name} was already loaded')
            
            self.click = False
            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == KEYDOWN:
                    if event.key == K_ESCAPE:
                        pygame.quit()
                        sys.exit()
                if event.type == MOUSEBUTTONDOWN:
                    if event.button == 1:
                        self.click = True
                        
            pygame.display.update()
            
        self.model_name = model_name
        self.select_target_layer()
        
        if self.cam:
            try:
                print('Reloading CAM method with new Model')
                self.cam = self.load_cam()
            except:
                print('Some error ocurred, try loading cam to CPU')
                self.cam = self.load_cam(True)
        elif self.cam_name:
            try:
                print('Reloading CAM method with new Model')
                self.cam = self.load_cam(False, self.cam_name)
            except:
                print('Some error ocurred, try loading cam to CPU')
                self.cam = self.load_cam(True, self.cam_name)
        else:
            print('CAM method has not been selected, press M to choose one')
                
        if self.use_cuda:
            self.model.to('cuda')
            print(f"Selected model {model_name} is cuda ready")

        return model_name

    # img is a surface 
    def run_model(self, img):
        with torch.no_grad():
            preprocessed_image = pygame.surfarray.pixels3d(img)                
            preprocess = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ])
            if parameters.debug:
                preprocess_pil = Image.fromarray(np.uint8(preprocessed_image))
                preprocess_pil.show()
                input("wait for user input to pass preprocessed image")
            #Image.fromarray is rotating the image
            input_tensor = Image.fromarray(np.uint8(preprocessed_image)).convert('RGB')
                
            input_tensor = preprocess(input_tensor)
            input_tensor = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

            if self.use_cuda:
                input_tensor = input_tensor.to('cuda')
                
            output = self.model(input_tensor)
            return output
    
    #This option obtains the inference results from outside the cam method
    def get_detection(self, img):
        output = self.run_model(img)
        if output.size() == torch.Size([1, 1000]):
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            probabilities = probabilities.to('cpu')
            target_class = np.argmax(probabilities.detach().numpy())
            class_name = self.class_list[target_class]
            class_score = probabilities[target_class]
            if parameters.debug:
                print(f'target class is {target_class}')
                print(f"SINGLE DETECTION: {class_name} || {class_score*100}% ")
                
            return class_name, class_score.cpu().detach().numpy(), probabilities
        else:
            print(f'Not ImageNet:{output.size()}')

    
    
    def get_top_detections(self, input_image = None, probabilities = None, num_detections = 5):
        if probabilities is None:
            if input_image is not None:
                try:
                    _,_,probabilities = self.get_detection(input_image)
                    probabilities = probabilities.to('cpu')
                    probabilities = probabilities.cpu().detach().numpy()
                    top_locations = np.argpartition(probabilities, -num_detections)[-num_detections:]
                    ordered_locations = top_locations[np.argsort((-probabilities)[top_locations])]
                    np.flip(ordered_locations)
                    print(f'Top {num_detections} ordered results:')
                    ordered_score_percentages = []
                    for pos in ordered_locations:
                        class_name, class_percentage = self.get_class_and_score(probabilities, pos)
                        print(f"Class detected: {class_name} with score: {class_percentage}%")
                        ordered_score_percentages.append(class_percentage)
                    return ordered_locations, ordered_score_percentages
                
                except Exception as e:
                    print(
                        f'The ouput type was not expected:\n{e}')
            else:
                print('inputs missing')
                return None
    
    
    def prob_calc_efficient(self, output):
        # The output has unnormalized scores. To get probabilities, run a softmax on it.
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        probabilities = probabilities.to('cpu')
        target_class = np.argmax(probabilities.data.numpy())
        class_name = self.class_list[target_class]
        class_score = probabilities[target_class]
        return class_name, class_score.cpu().detach().numpy()
    
    
    def get_class_and_score(self, probabilities, index):
        class_name = self.class_list[index]
        class_score = probabilities[index]
        return class_name, round(class_score*100,2)

    
    def run_cam_filtered(self, img, cam_method = None, selected_location = None, new_method_name = None):
        if cam_method is None:
            cam_method = self.cam
        else:
            # try this to free GPU memory and avoid errors (cam must be instanced afterwards again!)
            del self.cam
            
        gc.collect()
        torch.cuda.empty_cache()
        t0 = time.time()
        # get the top detected classes to select a target later:
        top_detected_classes, top_detected_percentages = self.get_top_detections(img,  probabilities = None, num_detections = 10)
        self.target_class = roc_functions.check_relevant_classes(top_detected_classes, self.class_list)
        index = np.where(top_detected_classes == self.target_class)
        target_score_percent = top_detected_percentages[int(index[0])]
        # get the cam heat map in a pygame image
        surface, inf_outputs, _ =  roc_functions.surface_to_cam(
            img, cam_method, self.use_cuda, [ClassifierOutputTarget(self.target_class)])
        print('time needed for visualization method creation :', time.time()-t0)
        
        t1 = time.time()
        if inf_outputs is not None:
            class_name, class_score = self.prob_calc_efficient(inf_outputs)
        else:
            class_name, class_score, inf_outputs = self.get_detection(img)

        class_percentage = str(round(class_score*100,2))
        print('time needed for probabilities calculation:', time.time()-t1)
        
        print(f'In filtered CAM:\n',
              f'Class Filtered: {self.class_list[self.target_class]} | Class from CAM: {class_name}\n',
              f'Score Filtered: {target_score_percent} | Score from CAM: {class_percentage}\n',)
        
        if selected_location is None:
            self.surface = surface
            self.render_cam()
            # self.render_text()
        else:
            score_string = f"Class detected: {self.class_list[self.target_class]} with score: {target_score_percent}%"
            self.render_cam(selected_location, surface, score_string, new_method_name)
            # self.render_text()
        
        return self.class_list[self.target_class], target_score_percent
        
        
    def run_cam(self, img, cam_method = None, selected_location = None, new_method_name = None):
        if cam_method is None:
            cam_method = self.cam
        else:
            # try this to free GPU memory and avoid errors (cam must be instanced afterwards again!)
            del self.cam
            
        gc.collect()
        torch.cuda.empty_cache()
        t0 = time.time()
        # get the cam heat map in a pygame image
        surface, inf_outputs, cam_targets =  roc_functions.surface_to_cam(img, cam_method, self.use_cuda)
        print('time needed for visualization method creation :', time.time()-t0)
        
        t1 = time.time()
        if inf_outputs is not None:
            class_name, class_score = self.prob_calc_efficient(inf_outputs)
        else:
            class_name, class_score, inf_outputs = self.get_detection(img)

        class_percentage = str(round(class_score*100,2))
        print('time needed for probabilities calculation:', time.time()-t1)
        
        if selected_location is None:
            self.surface = surface
            self.render_cam()
        else:
            score_string = f"Class detected: {class_name} with score: {class_percentage}%"
            self.render_cam(selected_location, surface, score_string, new_method_name)
        
        return class_name, class_percentage
        
    # Compare methods while managing GPU memory usage to avoid errors
    def compare_new_method(self, img):
        new_method_name = self.select_cam(second_method = True)
        old_method_name = self.method_name
        new_cam_method = self.load_cam(method_name = new_method_name)
        if self.display_manager:
            self.display_manager.render()
        if self.filtered_cam:
            if self.display_manager is None:
                img_w, img_h = img.get_width(), img.get_height()
                print(f'image width is {img_w}, display width is {self.display_width}')
                self.compare_location = [self.display_width-img_w, 0]
                print(f'New location to compare is {self.compare_location}')
            else:
                self.compare_location = [int(2*self.display_width/3), int(self.display_height/2)]
            class_name, class_score = self.run_cam_filtered(img, new_cam_method, self.compare_location, new_method_name)
        else:
            class_name, class_score = self.run_cam(img, new_cam_method, self.compare_location, new_method_name)
        print(f"compared {old_method_name} to {new_method_name} \
            -> finished with output {class_name}|{class_score}%")
        time.sleep(10)
        # After execution it is necessary to free memory by deleting the second method
        del new_cam_method
        gc.collect()
        torch.cuda.empty_cache()
        # Reload initial method, deleted in load cam if to free GPU memory
        self.cam = self.load_cam(False, method_name = old_method_name)
        return class_name, class_score    
    
    
    def render_cam(self, selected_location = None, surface_to_plot = None, second_classification = None, new_method_name = None):
        if self.surface:
            self.display.blit(self.surface, self.main_location)
            self.text_render()

        if selected_location == self.compare_location:
            print('plotting the second CAM output...')
            self.display.blit(surface_to_plot, selected_location)
            pygame.display.update()
            if second_classification is not None and new_method_name is not None:
                self.text_render(second_classification, new_method_name)


    def text_render(self, second_classification = None, second_method_name = None):
        
        if second_method_name is not None:
            description = f'Model: {self.model_name} Method: {second_method_name}'
        else:
            description = f'Model: {self.model_name} Method: {self.cam_name}'
            
        description_text = self.font.render(description, True, (255, 255, 255))
        
        if second_classification is None:
            score_output = self.font.render(self.classification_output, True, (255, 255, 255))
            loc = self.main_location
        else:
            print(f'second classification is {second_classification}')
            score_output = self.font.render(second_classification, True, (255, 255, 255))
            loc = self.compare_location
        
        score_loc = [loc[0], loc[1] + 20]
        self.display.blit(description_text, loc)
        self.display.blit(score_output, score_loc)    
        pygame.display.update()
    
    
    def clear_memory(self):
        print('\n\nmemory before Model deletion:')
        print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
        print("torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(0)/1024/1024/1024))
        print("torch.cuda.max_memory_reserved: %fGB\n\n"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024))
        del self.model
        print('memory after Model deletion:')
        print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
        print("torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(0)/1024/1024/1024))
        print("torch.cuda.max_memory_reserved: %fGB\n\n"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024))
        if self.cam:
            print('\n\nmemory after CAM method deletion:')
            print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
            print("torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(0)/1024/1024/1024))
            print("torch.cuda.max_memory_reserved: %fGB\n\n"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024)) 
            del self.cam
            print('memory before CAM method deletion:')
            print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
            print("torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(0)/1024/1024/1024))
            print("torch.cuda.max_memory_reserved: %fGB\n\n"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024))
        
        gc.collect()
        torch.cuda.empty_cache()
        self.model = None
        self.cam = None
    
        
    def run_menu_no_loop(self, event, call_exit, input_image, offset):
        if not self.main_location:
            self.main_location = offset
        if parameters.paused == True:
            last_pause = True
            if not last_pause:
                print('PAUSED')
        else:
            last_pause = False
        
        call_exit = False          
        if event.type == pygame.QUIT:
            call_exit = True
            return call_exit
        elif event.type == pygame.KEYDOWN:
            if event.key == K_ESCAPE or event.key == K_q:
                call_exit = True
                return call_exit
            elif event.key == K_SPACE:
                parameters.paused = not parameters.paused
                if parameters.paused:
                    if self.cam is not None:
                        if input_image:
                            if self.filtered_cam:
                                class_name, class_score = self.run_cam_filtered(input_image)
                            else:
                                class_name, class_score = self.run_cam(input_image)
                            self.last_image_evaluated = input_image
                        else:
                            print('[W] No input image')
                            if self.filtered_cam:
                                class_name, class_score = self.run_cam_filtered()
                            else:
                                class_name, class_score = self.run_cam()
                        
                        self.classification_output = f"Class detected: {class_name} with score: {class_score}%"
                        print(self.classification_output)
                    else:
                        no_cam_warning = "CAM method is not selected, Press button M"
                        print(no_cam_warning)
                        draw_text(no_cam_warning, self.font, (255, 255, 255), self.display, 0, 0)
                        pygame.display.update()
                        parameters.paused = False
                        
            elif event.key == pygame.K_m:
                if not parameters.paused:
                    parameters.paused = True
                    cam_name = self.select_cam()
                    
                    if self.display_manager:
                        self.display_manager.render()
                    else:
                        roc_functions.blip_image_centered(self.display, input_image)
                    
                    if cam_name != self.cam_name:
                        self.cam = self.load_cam()
                        self.cam_name = cam_name
                        cam_selected = (f'{cam_name} selected, loading...')
                        print(cam_selected)
                        pygame.display.update()
                    else:                        
                        print(f'{cam_name} selected, loaded')
                    
                    parameters.paused = False
                    return False
                else:
                    print('Comparing with another method')
                    if input_image:
                            class_name, class_score = self.compare_new_method(input_image)
                    else:
                        print('[W] No input image')
                
            elif event.key == pygame.K_n:
                if not parameters.paused:
                    parameters.paused = True
                    self.select_model()
                    
                    if self.display_manager:
                        self.display_manager.render()
                    else:
                        roc_functions.blip_image_centered(self.display, input_image)
                    
                    parameters.paused = False
                    return False
            
            elif event.key == pygame.K_t:
                self.get_top_detections(input_image)
                return False

            elif event.key == pygame.K_o:
                self.get_detection(input_image)
                return False
                    
                
        if parameters.paused:
            self.render_cam(offset)
            pygame.display.update()     
                            
if __name__ == '__main__':
    pygame.init()
    pygame.font.init() #for fonts rendering
    display = pygame.display.set_mode([1920,1080], pygame.HWSURFACE | pygame.DOUBLEBUF)
    test_menu = gui_CAM(display)
    call_exit = False
    path = '/home/roc/imagenet-sample-images'

    try:
        image_name = random.choice(os.listdir(path))
    except Exception as e:
        print(f'Selected folder path {path} is not correct\n')
        path = 'utils/test_images/carla_input'
        image_name = random.choice(os.listdir(path))
    print(image_name)
    file_path = os.path.join(path, image_name)    

    sample_image = pygame.image.load(file_path)
    display.blit(sample_image, [0,0])
    roc_functions.blip_image_centered(display, sample_image)
    while not call_exit:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    image_name = random.choice(os.listdir(path))
                    file_path = os.path.join(path, image_name)
                    sample_image = pygame.image.load(file_path)
                    parameters.paused = False
                    roc_functions.blip_image_centered(display, sample_image)
                    
            call_exit = test_menu.run_menu_no_loop(event, call_exit, sample_image, [0,0])
