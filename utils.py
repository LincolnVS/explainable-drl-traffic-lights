import numpy as np
import pandas as pd
import csv
import os
import json
  
# Opening JSON file
def get_info_file(file_name):
    try:
        data = {}
        with open(file_name) as json_file:
            data = json.load(json_file)

        print(data)
        assert data.get("validation",False) == True, "Validation Token not found"

    except Exception as e:
        print("Error while oppening json file with config")
        print(e)
        exit()

    return data
    

def append_new_line_states(file_name, lista):
    """Append given list as a new line at the end of file"""
    # Open the file in append & read mode ('a+')
    path = f'{file_name}_state.csv'
    list_to_save = []
    list_to_save.append(lista[0])
    list_to_save.append(lista[1])
    list_to_save.append(lista[2][0])
    list_to_save.append(lista[3][0])
    list_to_save.append(lista[4])
    list_to_save.append(lista[5])

    list_to_save = np.array(list_to_save, dtype="object").reshape(-1,len(list_to_save))
    df = pd.DataFrame(list_to_save, columns = ['episode','i','actual_state','next_state','Mphase','Iphase'])
    df.to_csv(path,sep=';', mode='a',index=False, header=not os.path.exists(path))
    
    #with open(file_name, "a+") as file_object:
        #file_object.write(list_to_append)
        #file_object.write("\n")
        #write = csv.writer(file_object, delimiter = ';')
        #print(list_to_append)
        #write.writerows(map(lambda x: [x], list_to_append))
        #file_object.close()

def append_new_line(file_name, lista):
    """Append given list as a new line at the end of file"""
    # Open the file in append & read mode ('a+')
    
    list_to_save = []
    list_to_save.append(lista[0][0])
    list_to_save.append(lista[0][1])
    list_to_save.append(lista[1])
    list_to_save.append(lista[2])
    list_to_save.append(lista[3][0])
    list_to_save.append(lista[3][1])
    list_to_save.append(lista[4])
    list_to_save.append(lista[5])

    list_to_save = np.array(list_to_save, dtype="object").reshape(-1,len(list_to_save))
    df = pd.DataFrame(list_to_save, columns = ['actual_state','actual_phase','action','reward','next_state','next_phase','episode','i'])
    df = df[['episode','i','actual_state','actual_phase','action','reward','next_state','next_phase']]
    df.to_csv(f'{file_name}.csv',sep=';', mode='a',index=False, header=not os.path.exists(f'{file_name}.csv'))
  