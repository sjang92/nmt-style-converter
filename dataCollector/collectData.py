import sys
import time
from datetime import datetime

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import selenium

import Tkinter as tk

from xvfbwrapper import Xvfb

from AutomatorError import AutomatorSetupError
from selenium.common.exceptions import ElementNotVisibleException


class Translate_Automator(object):
    """ A translate-automator automates the data collecting job from
    Google Translate. It should be initialized with the following properties:

    Attributes:
        driver: A Selenium driver of desired browser. e.g. webdriver.Firefox()
        user: A client's gmail username.
        pw: A client's gmail password
    """

    GMAIL_URL = "http://gmail.com"
    TRANSLATE_URL = "https://translate.google.com/"

    SOURCE_LANGUAGE_BUTTON = "gt-sl-gms"
    RESULT_LANGUAGE_BUTTON = "gt-tl-gms"

    SOURCE_LANGUAGE_MENU = "gt-sl-gms-menu"
    RESULT_LANGUAGE_MENU = "gt-tl-gms-menu"

    LANGUAGE_ITEM = "goog-menuitem-content"

    SOURCE_BOX = "source"

    RESULT_EDIT_BOX = "gt-res-edit"
    RESULT_CONTRIB_BUTTON = "contribute-target"

    def __init__(self, driver=webdriver.Firefox(), user="", pw="", fileName=""):
        self.vdisplay = Xvfb()
        self.vdisplay.start()
        self.root = tk.Tk()
        self.driver = driver
        self.user = user
        self.pw = pw
        self.fileName = fileName
        self.content = []
        self.result = []

    def setup(self):
        if self.user == "" or self.pw == "":
            raise AutomatorSetupError(self.driver, self.user, self.pw)

        self.driver.get(self.GMAIL_URL)

        self.driver.find_element_by_id('Email').send_keys(self.user)
        self.driver.find_element_by_id('next').click()
        time.sleep(2)

        self.driver.find_element_by_id('Passwd').send_keys(self.pw)
        self.driver.find_element_by_id('signIn').click()
        time.sleep(2)

        self.driver.get(self.TRANSLATE_URL)
        time.sleep(2)

        print "Automator: %s, Setup Complete" % self

    def __str__(self):
        return "Driver: %s, user: %s, password %s" % (self.driver, self.user, self.pw)

    def readFile(self):
        with open(self.fileName) as f:
            _content = f.readlines()
        _content = [line.decode("utf-8").encode("ascii","ignore").strip() for line in _content]

        self.content = _content

    def run_batch(self):
        self.readFile()
        for line in self.content:
            self.run_step(line)

    def close(self):
        self.driver.close
        self._writeResultToFile()

        self.vdisplay.stop()

    def run_step(self, sentence):
        self._putEnglishToSourceBox(sentence)

        self._setLanguage("result", "Japanese")

        self._copyResultFromResultBoxToSourceBox()

        self._setLanguage("source", "Japanese")
        self._setLanguage("result", "English")

        self._copyResultFromResultBoxToClipBoard()

        self.result.append(self.root.clipboard_get())

        self._clearBoxes()

    def _setLanguage(self, boxType, targetLang):
        languageList = []

        if boxType == "source":
            self.driver.find_element_by_id(self.SOURCE_LANGUAGE_BUTTON).click()
            languageList = self.driver.find_element_by_id(self.SOURCE_LANGUAGE_MENU).find_elements_by_class_name(self.LANGUAGE_ITEM)
        if boxType == "result":
            self.driver.find_element_by_id(self.RESULT_LANGUAGE_BUTTON).click()
            languageList = self.driver.find_element_by_id(self.RESULT_LANGUAGE_MENU).find_elements_by_class_name(self.LANGUAGE_ITEM)
        time.sleep(1)

        for language in languageList:
            if language.text == targetLang:
                language.click()
                break

        time.sleep(1)

    def _clearBoxes(self):
        self.driver.find_element_by_id(self.SOURCE_BOX).clear()
        self.driver.find_element_by_id(self.RESULT_CONTRIB_BUTTON).clear()
        time.sleep(1)

    def _putEnglishToSourceBox(self, sentence):
        self._setLanguage("source", "English")
        source = self.driver.find_element_by_id(self.SOURCE_BOX)
        source.clear()                #make sure source box is empty
        source.send_keys(sentence)    #put sentence into the source box
        source.send_keys(Keys.RETURN) #press Enter

    def _copyResultFromResultBoxToSourceBox(self):
        time.sleep(1)
        source = self.driver.find_element_by_id(self.SOURCE_BOX)

        copy = self.driver.find_element_by_id(self.RESULT_EDIT_BOX)
        copy.click()

        time.sleep(1)

        copy = self.driver.find_element_by_id(self.RESULT_CONTRIB_BUTTON)
        copy.click()

        copy.send_keys(Keys.COMMAND, 'a')
        copy.send_keys(Keys.COMMAND, 'c')
        time.sleep(1)

        source.click()
        source.send_keys(Keys.COMMAND, 'a')
        source.send_keys(Keys.COMMAND, 'v')
        time.sleep(1)

    def _copyResultFromResultBoxToClipBoard(self):
        time.sleep(1)
        copy = self.driver.find_element_by_id(self.RESULT_EDIT_BOX)
        copy.click()
        time.sleep(1)

        copy = self.driver.find_element_by_id(self.RESULT_CONTRIB_BUTTON)
        copy.click()

        copy.send_keys(Keys.COMMAND, 'a')
        copy.send_keys(Keys.COMMAND, 'c')
        time.sleep(1)

    def _writeResultToFile(self):
        resFile = open("results/"+str(datetime.now())+".result", "w")
        for item in self.result:
            resFile.write("%s\n" % item)


if __name__ == "__main__":
    start = time.time()
    a = Translate_Automator(webdriver.Firefox(), sys.argv[1], sys.argv[2], sys.argv[3])
    a.setup()
    a.run_batch()
    a.close()
    end = time.time()
    print "Running time: %s sec\n" % str(end-start)
