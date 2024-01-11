import re

def Clean_data(data):
    """Removes all the unnecessary patterns and cleans the data to get a good sentence"""
    repl='' #String for replacement
    
    #removing all open brackets
    data=re.sub('\(', repl, data)
    
    #removing all closed brackets
    data=re.sub('\)', repl, data)
    
    #Removing all the headings in data
    for pattern in set(re.findall("=.*=",data)):
        data=re.sub(pattern, repl, data)
    
    #Removing unknown words in data
    for pattern in set(re.findall("<unk>",data)):
        data=re.sub(pattern,repl,data)
    
    #Removing all the non-alphanumerical characters
    for pattern in set(re.findall(r"[^\w ]", data)):
        repl=''
        if pattern=='-':
            repl=' '
        #Retaining period, apostrophe
        if pattern!='.' and pattern!="\'":
            data=re.sub("\\"+pattern, repl, data)
            
    return data
