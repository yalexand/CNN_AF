# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 18:05:29 2022

@author: alexany
"""

from CNN_AF import stack_data_trainer, stack_data_tester

modes = ['proj','image']
MobileNets = ['V2','V3']

# for m in modes:
#     for n in MobileNets:
#         #
#         sdtr = stack_data_trainer(mode = m, MobileNet = n)       
#         sdtr.set_data_folder("c:\\users\\alexany\\tmp\\2022-03-09-18-19\\")
#         sdtr.batch_size = 67
#         sdtr.epos = 240   
#         sdtr.train(validation_fraction = 0.1)
#         #
#         sdts = stack_data_tester(mode = m, MobileNet = n)
#         sdts.set_data_folder("c:\\users\\alexany\\tmp\\2022-03-09-18-19\\")
#         sdts.test(zStep=0.05, 
#                   validation_fraction = 0.1,
#                   results_folder = sdtr.results_folder + '\\',
#                   token = m + '_' + n,
#                   )

N =[8000,6000]
for num_classes in N:
        #
        sdtr = stack_data_trainer('proj', MobileNet = 'V3')       
        sdtr.set_data_folder("c:\\users\\alexany\\tmp\\2022-03-09-18-19\\")
        sdtr.batch_size = 67
        sdtr.epos = 300   
        sdtr.MobileNetV3_num_classes = num_classes
        sdtr.train(validation_fraction = 0.1)
        #
        sdts = stack_data_tester('proj', MobileNet = 'V3')
        sdts.set_data_folder("c:\\users\\alexany\\tmp\\2022-03-09-18-19\\")
        sdts.MobileNetV3_num_classes = num_classes
        sdts.test(zStep=0.05, 
                  validation_fraction = 0.1,
                  results_folder = sdtr.results_folder + '\\',
                  token = 'n_classes_' + str(sdtr.MobileNetV3_num_classes) + '_epos_' + str(sdtr.epos),
                  )

