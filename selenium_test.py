import time

from selenium import webdriver
from selenium.webdriver.common.keys import Keys

from AutomatorError import AutomatorSetupError


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

    LANGUAGE_BUTTON = "gt-tl-gms"
    LANGUAGE_JA = ":47"
    LANGUAGE_ENG = ":3j"

    SOURCE_BOX = "source"
    RESULT_EDIT_BOX = "gt-res-edit"
    RESULT_CONTRIB_BUTTON = "contribute-target"

    def __init__(self, driver=webdriver.Firefox(), user="", pw=""):
        """
        """
        self.driver = driver
        self.user = user
        self.pw = pw

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

        print "Automator: %s, Setup Complete" % self

    def __str__(self):
        return "Driver: %s, user: %s, password %s" % (self.driver, self.user, self.pw)

    def run_step(self, sentence):

        self.driver.get(self.TRANSLATE_URL)

        self.driver.find_element_by_id(self.LANGUAGE_BUTTON).click()
        self.driver.find_element_by_id(self.LANGUAGE_JA).click()

        source = self.driver.find_element_by_id(self.SOURCE_BOX)
        source.clear()
        source.send_keys(sentence)
        source.send_keys(Keys.RETURN)
        time.sleep(2)

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

        self.driver.find_element_by_id(self.LANGUAGE_BUTTON).click()
        self.driver.find_element_by_id(self.LANGUAGE_ENG).click()

    def close(self):
        self.driver.close()


a = Translate_Automator(webdriver.Firefox(), "", "")
a.setup()
a.run_step("We the People of the United States, in Order to form a more perfect Union, establish Justice, insure domestic Tranquility, provide for the common defence, promote the general Welfare, and secure the Blessings of Liberty to ourselves and our Posterity, do ordain and establish this Constitution for the United States of America.")
a.close()
