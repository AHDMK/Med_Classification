def Abstract_extract(path_in,path_out):
    with open(path_in,'r',encoding="utf8") as infile, open(path_out, 'w',encoding="utf8") as outfile:
        copy = False
        for line in infile:
        
            if line.startswith("AB  -"):
                copy = True
                x=0
            elif line.startswith("FAU - ") or line.startswith("CI  -") or line.startswith("CN  -") or line.startswith("PG - ") or line.startswith("LA  - "):
                copy = False
                if x==0:
                    outfile.write("\n")
                    outfile.write("\n")
                x+=1
            if copy:
                line = line.rstrip()
                line = line.replace('\n',' ')
                line = line.replace('\"',' ')
                outfile.write(line.replace('      ',' '))
                
def Categorization(path_in,path_out,Category) :  
    Category = Category + ',\"'           
    with open(path_in, 'r',encoding="utf8") as infile, open(path_out, 'w',encoding="utf8") as outfile:
        outfile.write("Category,text\n") 
        for line in infile:
            if line.startswith("AB  -") and line != 'AB  -':
                line = line.split()
                line.append('"')
                line.pop(1)
                line[0] = Category
                line = ' '.join(line)
                outfile.write(line)
                outfile.write('\n')
Abstract_extract("C:\Dataset/onc_da.txt","C:\Dataset/testing_data.txt")
Categorization("C:\Dataset/testing_data.txt","C:\Dataset/oncol_abstract.txt","Oncologist")
