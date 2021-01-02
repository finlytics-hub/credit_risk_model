import os 
import pandas as pd

path2bin = "/home/mitiku/Projects/client/Tesfaye/GermanCredit2/notebooks/working-dir/data.bin"

data = pd.read_csv(path2bin)
output = '<form class="form-inline">\n'

        
            
          
       

for col in data.columns[:-1]:
    col_words = col.split(".")
    col_words[0] = col_words[0][0].upper() + col_words[0][1:]
    new_col = " ".join(col_words)
    output+="""
    <div class="form-group col-lg-6">
    <label class="col-lg-6 form-control-label" for=\"""" 
    output +=  col+  '">'+new_col+"</label>\n"
    output += '<div class="col">\n'
    output += '<select class="form-control" id="{}" name = "{}">\n'.format(col, col)
    
    for val in data[col].unique():
        val = str(val)
        output += '<option value="'+ val+'">'+val+'</option>\n'
          
    output+= "</select>\n"  
    output+="</div>\n"  
    output+="</div>\n"
   
  
        
output+='<button type="submit">\n'
output+='Submit\n'
output+="</button>\n"
output+="</form>"
print(output)