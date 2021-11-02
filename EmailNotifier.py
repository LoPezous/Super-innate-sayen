import datetime
import win32com.client as win32

try:

    start = datetime.datetime.now().time()
    UMAP_clusters(r'metrics/euclidean',
                  5000,
                  10,
                  'euclidean')
    finish = datetime.datetime.now().time()
    delta = datetime.timedelta(hours=finish.hour - start.hour, minutes=finish.minute - start.minute,
                               seconds=finish.second - start.second)

    for adress in ['martin.pezous@cea.fr', 'martin.pezous-puech@live.fr']:
        outlook = win32.Dispatch('Outlook.Application')
        mail = outlook.CreateItem(0)
        mail.To = adress
        mail.Subject = 'Analysis: Done'
        mail.Body = ''
        mail.HTMLBody = 'The analysis was succesful and ran for ' + str(delta) + ' (Days : Hours : Minutes : seconds)'
        mail.Send()


except Exception as e:

    for adress in ['martin.pezous@cea.fr', 'martin.pezous-puech@live.fr']:
        outlook = win32.Dispatch('Outlook.Application')
        mail = outlook.CreateItem(0)
        mail.To = adress
        mail.Subject = 'ERROR'
        mail.Body = ''
        mail.HTMLBody = 'An error occured during analysis:\n' + str(e) + '\n' + 'traceback : ' + traceback.format_exc()
        mail.Send()
