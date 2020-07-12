import selenium
from selenium import webdriver
from selenium.webdriver.common.keys import Keys  
import time
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
#使用chrome的webdriver
driver = webdriver.Chrome()
 
driver.get('https://www.instagram.com/p/B-cEjCFja0H/?utm_source=ig_web_copy_link/')

#*[contains(@id,'query')
homebtn = driver.find_element_by_class_name("sqdOP").click()
time.sleep(5)
followerbtn = driver.find_element_by_class_name("-nal3").click()
time.sleep(2)
elem_user = driver.find_element_by_name("username")
elem_user.clear()
elem_user.send_keys("_lanlan1996_")
elem_pwd = driver.find_element_by_name("password")
elem_pwd.clear()
elem_pwd.send_keys("linda0404")  
elem_pwd.send_keys(Keys.RETURN)
time.sleep(5)
#followerbtn2 = driver.find_element_by_xpath("//a[@href='/nini_foodaworld/followers/']").click()
WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, "//a[@href='/nini_foodaworld/followers/']"))).click()

time.sleep(2)
follower = driver.find_elements_by_xpath("//div[contains(@class,'PZuss')]")
time.sleep(5)
for i in range(100):
    time.sleep(2)
    targetElem = driver.find_element_by_xpath("//div[contains(@class,'d7ByH')]")
    targetElem.send_keys(Keys.TAB)
for i in range(len(follower)):
    print(follower[i].get_attribute('textContent'))
'''
for i in range(len(follower)):
    WebDriverWait(driver, 5).until(ExpectedConditions.visibilityOfElementLocated(By.id("document-content")));
    WebElement element = driver.findElement(By.xpath("///div[contains(@class,'PZuss')]/*[last()]")); 
    ((JavascriptExecutor) driver).executeScript("arguments[0].scrollIntoView(true);", element);

    #target = driver.find_element_by_xpath("//div[contains(@class,'isgrP')]")
    #driver.execute_script("arguments[0].scrollIntoView();", target)
    #driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    #driver.execute_script("window.scrollBy(0,200)", "") 
    #target = driver.find_elements_by_xpath("//div[@role='dialog']")
    #target = driver.find_element_by_class_name("PZuss")  
    #driver.execute_script("arguments[0].scrollIntoView();", target)  
    #将网页滚动到网页底部
    #driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    #driver.execute_script("window.scrollTo(0, document.body.scrollHeight);", target)
    # 向下滚动200个像素，鼠标位置也跟着变了
    #driver.execute_script("window.scrollBy(0,200)", "") 

    print(follower[i].get_attribute('textContent'))
#find the followers window
#target = driver.find_element_by_xpath("//div[contains(@class,'isgrP')]")
#driver.execute_script("arguments[0].scrollIntoView();", target)
#print(target)
'''
print("结束拖动滚动条....")
time.sleep(2)
driver.close()