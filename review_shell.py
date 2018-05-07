
# coding: utf-8

# # Amazon Fine Food Reviews Sentiment Analysis with Recurrent Neural Network

# In[37]:


import pandas as pd
import numpy as np
from collections import namedtuple
import tensorflow as tf
from string import punctuation
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.metrics import roc_curve, auc, classification_report

sns.set(color_codes=True)

import sys,cmd

latestFileName = '../Zomato-Scraper/zomato_chandigarh_2.json';

restaurantsData = open(latestFileName , 'r');

class RestaurantReviewShell(cmd.Cmd):

    intro = 'Welcome to the restaurant reviews shell.   Type help or ? to list commands.\n'
    prompt = '(analyze-it) '


    # ----- basic turtle commands -----
    def do_detail(self, arg):
        'Get detail for restaurant - name'
        self.get_details(arg);
  
    def do_bye(self, arg):
        'Stop recording, close the turtle window, and exit:  BYE'
        print('Thank you for using Turtle')
        self.close()
        bye()
        return True

    
    def precmd(self, line):
        line = line.lower()
        return line



    # basic app operations

    def check_exists

    def get_details(name):




if __name__ == '__main__':
    RestaurantReviewShell().cmdloop()