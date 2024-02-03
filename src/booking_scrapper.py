from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException, ElementClickInterceptedException, StaleElementReferenceException

import os
import re
import time
import requests
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

class Booking_Scrapper:

    def __init__(self, geko_path = None, booking_url = None ,profile_path = None):
        '''
        Constructor for the class
        '''
        self.geko_path = geko_path
        self.profile_path = profile_path
        self.headers = headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/75.0.3770.142 Safari/537.36'}
        if booking_url != None:
            self.booking_url = booking_url
        else:
            self.booking_url = 'https://www.booking.com/index.en-gb.html'

    def start_up(self, link, geko_path, profile_path=None, browser=None):
        '''
        Function to set up the browser and open the selected link (usin)
        profile_path: path to the profile to be used (if None, a new profile is created)
        '''
        if browser != None:
            browser.get(link)
            time.sleep(2)
            return browser
        else:
            if profile_path:
                firefox_options = webdriver.FirefoxOptions()
                firefox_options.add_argument(f'--profile={profile_path}')
                service = Service(geko_path)
                browser = webdriver.Firefox(service=service, options=firefox_options)
            
            else:
                profile = webdriver.FirefoxProfile()
                options = Options()
                options.profile = profile
                service = Service(geko_path)
                browser = webdriver.Firefox(service=service, options=options)
            
            
            # Website address here
            browser.get(link)
            time.sleep(2)  # Adjust sleep time as needed
            
            # Maximize the browser window for full screen
            browser.maximize_window()
            
            if not profile_path:
                # Click on "Accept cookies"
                browser.find_element(by='xpath',value='//button[@id="onetrust-accept-btn-handler"]').click()
            
            return browser
    
    
    def check_and_click(self, browser, path, type):
        '''
        Function that checks whether the object is clickable and, if so, clicks on
        it. If not, waits one second and tries again. After 3 seconds of waiting, it doesn't try anymore.
        '''
        start_time = time.time()  # Record the start time

        while True:
            try:
                if type == "xpath":
                    browser.find_element('xpath',path).click()
                    return "Clicked!"  # Element found and clicked successfully
                elif type == "css":
                    browser.find_element('css selector', path).click() 
                    return "Clicked!"  # Element found and clicked successfully
            except NoSuchElementException:
                pass  # Continue if element not found
            except Exception as e:
                print(f"An error occurred: {e}")
                return False  # Other unexpected errors

            time.sleep(1)
            elapsed_time = time.time() - start_time
            if elapsed_time >= 3:
                # print("* The element was not found in the page. *")
                return None            
            
            
    def check_obscures(self, browser, xpath, type):
        '''
        Function that checks whether the object is being "obscured" by any element so
        that it is not clickable. Important: if True, the object is going to be clicked!
        '''
        try:
            if type == "xpath":
                browser.find_element('xpath',xpath).click()
            elif type == "id":
                browser.find_element('id',xpath).click()
            elif type == "css":
                browser.find_element('css selector',xpath).click()
            elif type == "class":
                browser.find_element('class name',xpath).click()
            elif type == "link":
                browser.find_element('link text',xpath).click()
        except (ElementClickInterceptedException, NoSuchElementException, StaleElementReferenceException) as e:
            print(e)
            return False
        return True
    
    def select_place(self, browser, place):
        '''
        Function to choose a place of interest
        '''
        search1 = browser.find_element(by='xpath',value='//*[@id=":re:"]')
        search1.send_keys(place)
        time.sleep(2)

    def select_place(self, browser, place):
        '''
        Function to choose a place of interest
        '''
        # Find the search box
        search_box = browser.find_element(by='xpath', value='//*[@id=":re:"]')

        # Check if the search box is not empty
        if search_box.get_attribute("value"):
            search_box.clear()

        # Type the new place
        search_box.send_keys(place)
        time.sleep(2)

    def check_date_format(self, date_str):
        '''
        Make sure the dates are in the right format for the booking webpage
        '''
        pattern = re.compile(r'^\d{4}-\d{2}-\d{2}$')
        return bool(pattern.match(date_str))   
        
    def select_dates(self ,browser, start_date, end_date):
        '''
        Function to choose dates of interest
        '''
        # Get months of start_date and end_date
        s_d = datetime.strptime(start_date, '%Y-%m-%d')
        e_d = datetime.strptime(end_date, '%Y-%m-%d')
        start_month = s_d.strftime("%B %Y")
        end_month = e_d.strftime("%B %Y")

        # Click on the dates filter
        css='button.ebbedaf8ac:nth-child(2) > span:nth-child(1)'
        browser.find_element('css selector',css).click()
        
        # Open dates filter on the month of start_date
        path_month = '//button[@class="a83ed08757 c21c56c305 f38b6daa18 d691166b09 f671049264 deab83296e f4552b6561 dc72a8413c f073249358"]'
        wait = WebDriverWait(browser, 10)
        wait.until(EC.presence_of_element_located((By.XPATH, '//h3[@class="e1eebb6a1e ee7ec6b631"]')))

        while browser.find_element('xpath', '//h3[@class="e1eebb6a1e ee7ec6b631"]').text != start_month:
            browser.find_element('xpath', path_month).click()

        # Get the list of all dates
        path_dates = '//div[@id="calendar-searchboxdatepicker"]//table[@class="eb03f3f27f"]//tbody//td[@class="b80d5adb18"]//span[@class="cf06f772fa"]'
        dates = wait.until(EC.presence_of_all_elements_located((By.XPATH, path_dates)))
        
        if s_d.month == e_d.month or s_d.month == e_d.month + 1:
            for date in dates:
                if date.get_attribute("data-date") == start_date:
                    date.click()
                if date.get_attribute("data-date") == end_date:
                    date.click()
                    break
        else:
            for date in dates:
                if date.get_attribute("data-date") == start_date:
                    date.click()
                    break
            
            # Open dates filter on the month of end_date
            while browser.find_element('xpath', '//h3[@class="e1eebb6a1e ee7ec6b631"]').text != end_month:
                browser.find_element('xpath', path_month).click()
                dates = wait.until(EC.presence_of_all_elements_located((By.XPATH, path_dates)))
                
                for date in dates:
                    if date.get_attribute("data-date") == end_date:
                        date.click()
                        break


    def get_number_pages(self, browser):
        '''
        Function to get total number of pages. 
        '''
        #a = browser.find_elements('xpath', '//button[@class="a83ed08757 a2028338ea"]')
        wait = WebDriverWait(browser, 10)
        a = wait.until(EC.presence_of_all_elements_located((By.XPATH, "//button[@class='a83ed08757 a2028338ea']")))
        return(int(a[-1].text))


    def get_hotels(self, browser):
        '''
        Function to get hotels from a page. Returns a pandas dataframe.
        '''
        temp_list = []
        # hotel_placeholder =  browser.find_elements('xpath','//div[@class="c066246e13"]')
        wait = WebDriverWait(browser, 10)
        hotel_placeholder = wait.until(EC.presence_of_all_elements_located((By.XPATH,'//div[@class="c066246e13"]')))
        
        for hotel in hotel_placeholder:
            hotel_name = hotel.find_element('xpath','.//div[@class="f6431b446c a15b38c233"]').text
            hotel_price = hotel.find_element('xpath','.//span[@class="f6431b446c fbfd7c1165 e84eb96b1f"]').text 
            hotel_link = hotel.find_element('xpath','.//a[@class="a78ca197d0"]').get_attribute('href')
        
            try:
                hotel_description = hotel.find_element('xpath','.//div[@class="abf093bdfe d323a31618"]').text
            except NoSuchElementException:
                hotel_description = np.nan
            
            try:
                hotel_rating = hotel.find_element('xpath','.//div[@class="a3b8729ab1 d86cee9b25"]').text   
            except NoSuchElementException:
                hotel_rating = np.nan
            
            temp_list.append({'Hotel_Name': hotel_name, 'Price': hotel_price,
                        'Hotel_Description_Short': hotel_description, 'Rating': hotel_rating,
                        'url': hotel_link})
        
            # print(hotel_name)
            
        return temp_list 

    def save_dataframe_to_csv(self, df, city, start_date, end_date, interval_days=None):
        # Convert city to a string if it's a list
        if isinstance(city, list):
            city_name = '_'.join(city)
        else:
            city_name = str(city)

        # Build the file name
        file_name_parts = [city_name, start_date, end_date]
        if interval_days is not None:
            file_name_parts.append(f"{interval_days}_days")
        
        file_name = '_'.join(file_name_parts) + '.csv'
        
        # Ensure the 'data' folder exists
        if not os.path.exists('./data'):
            os.makedirs('./data')

        # Save DataFrame to CSV
        df.to_csv(os.path.join('data', file_name), index=False)
