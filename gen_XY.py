# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 18:05:29 2022

@author: alexany
"""

from CNN_AF import stack_data_generator 

sdg = stack_data_generator()

sdg.zStep = 0.05  # in um
sdg.image_filename = 'MMStack_Pos1.ome.tif'        
sdg.stack_dir_prefix = 'stack'
sdg.image_size = 96        

pList = []
root = "t:\\live\\20220223_AF_For_Yuriy\\Original_CNN\\"
pList.append(root+"20200618_MobileNetV2_fine_AF_train_no_sat_0615_0616_0617_test_all_val_0618\\")
pList.append(root+"20200615_20um_50nm_AF_74.65mA_7ms\\")
pList.append(root+"20200616_20um_50nm_AF_74.65mA_1.5ms\\")
pList.append(root+"20200617_20um_50nm_AF_74.66mA_5.5ms_NS\\") 

if not sdg.set_data_info(pList): 
    print('error!')
    quit()
    
sdg.set_output_folder_in("c:\\users\\alexany\\tmp\\")
sdg.gen_XY_in_range(0,len(sdg.pList))
