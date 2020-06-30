from networks.create_model import create_full_model,create_semi_model


CLASS_NAMES_DICTS={ 'hip_onfh':['Normal','ONFH_I','ONFH_II'],
                    'hip_3cls':['Normal','OA','ONFH'],
                    'hip_4cls':['Normal','OA','ONFH','DDH'],
                    'diabetes':['level_0','level_1','level_2','level_3','level_4'],
                    'skin':[ 'Melanoma', 'Melanocytic nevus', 'Basal cell carcinoma', 'Actinic keratosis',
                            'Benign keratosis', 'Dermatofibroma', 'Vascular lesion'],
                    'chest':['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 
                            'Pneumonia', 'Pneumothorax','Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 
                            'Pleural_Thickening', 'Hernia']}
CLASS_NUM_DICTS={'hip_onfh':[1066.0,259.0,659.0],
                  'hip_3cls':[6046.0,6906.0,9832.0],
                  'hip_4cls':[6046.0,5793.0,9832.0,4404.0],
                  'diabetes':[25802.0,2438.0,5288.0,872.0,708.0],
                  'skin':[1113.0, 6705.0, 514.0, 327.0, 1099.0, 115.0, 142.0],
                  'chest':[11559.0, 2776.0, 13317.0, 19894.0, 5782.0, 6331.0, 
                  1431.0, 5302.0, 4667.0, 2303.0,2516.0, 1686.0, 3385.0, 227.0]}
RESIZE_DICTS={'hip_onfh':256,
               'hip_3cls':256,
               'hip_4cls':256,
               'diabetes':384,
               'skin':224,
               'chest':384}
CREATE_MODEL_DICTS={'semi':create_semi_model,
                    'full':create_full_model}