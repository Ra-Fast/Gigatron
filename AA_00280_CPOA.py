from selenium import webdriver
from selenium.webdriver.support.ui import Select
import time
import os
import shutil
import pyautogui
import re
import win32com
import win32com.client
import datetime
import glob



def remove_email():
    outlook = win32com.client.Dispatch("Outlook.Application").GetNamespace("MAPI")
    #folder = outlook.Folders.Item("ProgramOffice RPA")
    #inbox = folder.Folders.Item("Inbox")
    inbox = outlook.GetDefaultFolder(6)
    messages = inbox.Items

    subject_filter="CPOA: Finished report."
    for mens in reversed(messages):
        # Get subject of email
        subject=mens.Subject

        # First step, filter by subject. It happens an exception with date for RECALL or UNDERIVERABLE
        # Filtering by subject will solve that

        if subject_filter in subject:
            mens.Delete()

def download_email():
    outlook = win32com.client.Dispatch("Outlook.Application").GetNamespace("MAPI")
    #folder = outlook.Folders.Item("ProgramOffice RPA")
    #inbox = folder.Folders.Item("Inbox")
    inbox = outlook.GetDefaultFolder(6)
    messages = inbox.Items
    regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    subject_filter="CPOA: Finished report."
    for mens in reversed(messages):
        # Get subject of email
        subject=mens.Subject

        # First step, filter by subject. It happens an exception with date for RECALL or UNDERIVERABLE
        # Filtering by subject will solve that

        if subject_filter in subject:
            match = re.findall(regex, mens.Body)
            match=match[0][:-1] # Remove last '>'
            return match
            
def download_report(url, driverpath, path_dl, path_sharepoint):
    #PATH= r'C:\Users\erafpac\OneDrive - Ericsson AB\Mi disco duro\Proyectos\Python\Selenium\IEDriverServer.exe'
    driver=webdriver.Ie(executable_path=driverpath)
    driver.get(url)
    btn=driver.find_element_by_link_text("Download Link")
    hrefurl = btn.get_attribute("href")
    driver.get(hrefurl)
    #driver.find_element_by_link_text("Download Link").click()
    time.sleep(10)
    # Press Save
    pyautogui.keyDown('altleft')
    pyautogui.press('s')
    pyautogui.keyUp('altleft')
    print("descarga")
    # Quit
    time.sleep(20)
    pyautogui.keyDown('ctrl')
    pyautogui.press('f4')
    pyautogui.keyUp('ctrl')
    driver.quit()
    move_report(path_dl, path_sharepoint)



def CPOA(pathdriver):
           
    driver=webdriver.Ie(executable_path=pathdriver)
    driver.get("https://spainstore.internal.ericsson.com/CPOA/application/wicket/bookmarkable/gui.process.po.report.generic.GenericReportFilter?4")
    
    try:
        # Open webpage
       
        # Select CPM
        driver.find_element_by_name("idPm").click()
        Select(driver.find_element_by_name("idPm")).select_by_visible_text("Oscar Vega Gutierrez(EVEGOSC)")
        driver.find_element_by_name("idPm").click()

        '''
        # Purchase order Status. Select ACTIVE
        driver.find_element_by_id("id16").click()

        # Purchase order Status. Select CLOSED
        driver.find_element_by_id("id17").click()
        # # Purchase order Status. Deselect REMOVED BY THE CUSTOMER
        driver.find_element_by_name("statusListContainer:finalStatusContainer:finalStatus:5:checkBox").click()
        '''
        # Search
        driver.find_element_by_name("btnOK").click()

        # Generate Reports
        driver.find_element_by_xpath("(//a[contains(text(),'Click here')])[2]").click()

       
    finally: 
        driver.quit()

        
def read_email(subject_filter, path):
    # Outlook handling
    outlook = win32com.client.Dispatch("Outlook.Application").GetNamespace("MAPI")

    folder = outlook.Folders.Item("ProgramOffice RPA")
    inbox = folder.Folders.Item("Inbox")
    messages = inbox.Items

    # Parameters
    #today=datetime.date.today().strftime("%Y-%m-%d")
    #today_2=datetime.date.today().strftime("%d-%m-%Y")
    found=False

     
             
    # Loop to search for the last email in mailbox
    for mens in reversed(messages):
        # Get subject of email
        subject=mens.Subject

        
        if found==False and subject_filter in subject:  # Filter by the subject to avoid issues with underiverables and recalls
            # Copy attachment in path
            for attachment in mens.Attachments:
                mail_extension=attachment.filename[-4:]
                if mail_extension in [".xls", "xlsx"]:
                    path_file=os.path.join(path, str(attachment.filename))
                    attachment.SaveAsFile(path_file)
                                
                    found=True
                    mens.Unread = True
                    break
            break
    return path_file    

def fecha():
    year = datetime.date.today().strftime("%Y")
    month = datetime.date.today().strftime("%m")
    day = datetime.date.today().strftime("%d")
    return str(year)+str(month)+str(day)

def move_report(path_dl, path_sharepoint):
    #path_dl=os.path.expanduser(r"C:\Users\erafpac\Downloads")
    #path_sharepoint=os.path.expanduser(r"C:\Users\erafpac\Ericsson\QF 3.0 - Solpro_Offer_Dump\CPOA_reports")
    all_files=os.path.join(path_dl,'*.xlsx')
    list_of_files = glob.glob(all_files) # * means all if need specific format then *.xlsx
    latest_file = max(list_of_files, key=os.path.getctime)
    filename=os.path.split(latest_file)[1]
    new_filename="CPOA_EVEGOSC_" + fecha()+".xlsx"
    os.rename(latest_file,os.path.join(path_dl, new_filename))
    shutil.copyfile(os.path.join(path_dl, new_filename),os.path.join(path_sharepoint,new_filename))

    os.remove(os.path.join(path_dl, new_filename))


if __name__ == "__main__":
    PATH= r'C:\Users\erafpac\OneDrive - Ericsson AB\Mi disco duro\Proyectos\Python\Selenium\IEDriverServer.exe'
    path_dl=os.path.expanduser(r"C:\Users\erafpac\Downloads")
    path_sharepoint=os.path.expanduser(r'C:\Users\erafpac\Ericsson\OSP Jumping Project Governance - Financial\AA_00280')
    
    print("CPOA")
    CPOA(PATH)
    time.sleep(300)
    download=download_email()
    download_report(download, PATH, path_dl, path_sharepoint)
    remove_email()
    
