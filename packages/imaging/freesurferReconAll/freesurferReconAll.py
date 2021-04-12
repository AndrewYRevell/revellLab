#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 28 15:10:18 2021

@author: arevell
"""

import os
import smtplib, ssl
import numpy as np
from glob import glob
from os.path import join, basename
from revellLab.packages.utilities import utils
from revellLab.packages.imaging.electrodeLocalization import electrodeLocalizationFunctions as elLoc
from email.mime.text import MIMEText
from email.header import Header
#%
BIDS = join("/media","arevell","sharedSSD","linux", "data", "BIDS")
dataset= "PIER"
outputDir = join(BIDS, "derivatives", "freesurferReconAll")
subDir = [join(BIDS,dataset, o) for o in os.listdir(join(BIDS,dataset))   if os.path.isdir(os.path.join(BIDS,dataset, o))]
subjects = [basename(item) for item in subDir ]

def does_match_in_array_of_string(key: str, search_list : list) -> bool:
   for item in search_list:
       if key in item:
           return True
   return False

def sendEmail(receiver_email = "andyrevell.python@gmail.com", subject ="Process is done", text = "done", port = 465, smtp_server = "smtp.gmail.com" , sender_email = "andyrevell.python@gmail.com"):
    
    AppPassword_twoAuthentication = "vgkpolyhiwzzifcc"
    email_message = MIMEText(text, 'plain', 'utf-8')
    email_message['From'] = sender_email
    email_message['To'] = receiver_email
    email_message['Subject'] = Header(subject, 'utf-8')
    context = ssl.create_default_context()
    with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
        server.login(sender_email, AppPassword_twoAuthentication)
        server.sendmail(sender_email, receiver_email, email_message.as_string())
    print(f"Email sent to {receiver_email}")

#%%
for s in range(len(subDir)):
    sub = basename(subDir[s])[4:]
    print(sub)
    acq = ["-3D_", "-T00_"] #acquisition description for the mprage. 
    sessions = glob(join(subDir[s], "*"))
    print(sessions)
    if len(sessions) > 0:
        for w in range(len(sessions)):
            session = basename(sessions[w])[4:]
            anats = np.array(glob(join(sessions[w], "anat", "*")))        
            T1s = anats[np.where(["_T1w.nii" in b for b in anats])[0]] #get T1w images
            if len(T1s) >0:
                #get the ones with specific acquisions that allow for proper freesurfer recon-all processing
                acquisition = [key for key in acq if does_match_in_array_of_string(key, T1s)]
                if len(acquisition) > 1:
                    IOError(f"More than 1 file exists with acq: \n{acq}") 
                else:
                    T1 = T1s[np.array([acquisition[0] in b for b in T1s])  ]
                    if len(T1) > 1:
                        IOError(f"More than 1 file exists with acq: \n{acquisition[0]}") 
                    else:
                        T1path = T1[0]
                        outputpath = join(outputDir, f"sub-{sub}", f"ses-{session}")
                        utils.checkPathAndMake(outputpath, outputpath)
                        elLoc.freesurferReconAll(T1path, outputpath,  overwrite = False, threads=12)

                        #send email done
                        sendEmail(subject = f"Recon-all {sub}: {session}" , text = f"{basename(T1path)}" )
   

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    