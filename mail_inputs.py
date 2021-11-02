import subprocess
import win32com.client
import time
def email_inputs():
    
    outlook = win32com.client.Dispatch('Outlook.Application')
    mapi = outlook.GetNamespace("MAPI")
    inbox = mapi.GetDefaultFolder(6)
        
    all_inbox=inbox.Items
    all_inbox = all_inbox.GetLast()
    found = False
    message_subject_to_find = 'inputz'
    subject_found = ''

    if all_inbox.Class == 43:
        if message_subject_to_find in all_inbox.Subject:
            subject_found = all_inbox.Subject
            found = True
            body = all_inbox.Body
            all_inbox.delete()
            return body


while True:
    c = email_inputs()
    if c != None:

        c = c.split(',')

        a_file = open(r"Serial.py", "r")

        list_of_lines = a_file.readlines()
        c[-1] = c[-1].replace('\r\n','')
        print(c)
        list_of_lines[454] = "                  " + c[0] + ","+'\n'
        list_of_lines[455] = "                  " + c[1] + ","+'\n'
        list_of_lines[456] = "                  " + '"' +c[2] + '"' +")"+"\n"

        a_file = open(r"Serial.py", "w")
        a_file.writelines(list_of_lines)
        a_file.close()
        a_file.close()

            #subprocess.run(r'cmd /k "set root=C:\Users\mp268043\Anaconda3"', shell = True)
            #subprocess.run(r'cmd /k "call %root%\Scripts\activate.bat %root%"', shell = True)
            #subprocess.run(r'cmd /k "Python C:\Users\mp268043\Desktop\Appl_like\Serial.py"', shell = True)
        #import Serial
        if c[3] == 'serial':
            exec(open("./Serial.py").read())
        if c[3] == 'parallel':
            exec(open("./Serial.py").read())
        else: 
            print(str(c[3]) + ' is not a valid method') 
    else:
        pass
