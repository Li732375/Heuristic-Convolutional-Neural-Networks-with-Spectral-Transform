import csv
from os import system

headers = ['model_id', 'parameter_count', 'accuracy'] #寫入欄位屬性

for model_id in range(31):
        with open('模型統計結果.csv','a', newline='') as model_csv:
                writeCsv = csv.writer(model_csv)
                fileString = None
                
                if model_id == 0: #寫入欄位屬性
                        writeCsv.writerow(headers)
                        continue

                try:
                        with open("model-" + str(model_id) + "/hillClimbing.log",
                                  "r") as file: #開啟個別記錄檔
                                parameter_count, accuracy = 0, 0
                                
                                fileString = file.readlines()[-1]
                                file.close()
                except FileNotFoundError:
                        print("\nThis model_id not exist！")
                        break
      
                print("====>\nopen", model_id, "final Line :", fileString)

                parameter_count_end = fileString.rfind(',')
                parameter_count_beg = fileString.rfind('parameter_count') + len('parameter_count') + 3
                accuracy_end = fileString.rfind('}')
                accuracy_beg = fileString.rfind('accuracy') + len('accuracy') + 3
                
                parameter_count = fileString[parameter_count_beg : parameter_count_end]
                accuracy = fileString[accuracy_beg : accuracy_end]
                  
                print('parameter_count ', parameter_count)
                print('accuracy ', accuracy)
                        
                writeCsv.writerow([str(model_id), parameter_count, accuracy])
                        
                model_csv.close()
print("\nFinish！")
system("pause")
