# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 14:55:59 2018

@author: Erik
"""

class sentence:
    def __init__(self, sentence, aspect_term, aspect_location, output_class):
        a = aspect(aspect_term, aspect_location, output_class)
        self.aspects = [a]
    
    def add_aspect(self, aspect_term, aspect_location, output_class):
        self.aspects.append(aspect(aspect_term, aspect_location, output_class))
        
class aspect:
    def __init__(self, aspect_term, aspect_location, output_class):
        self.aspect_term = aspect_term #use a list so we can append several aspect terms
        #split location into start and end locations
        self.aspect_start = int(aspect_location[:aspect_location.find("-")]) 
        self.aspect_end = int(aspect_location[aspect_location.rfind("-") + 1:])
        if "\n" in output_class:
            self.output_class = int(output_class[:-1])
        else:
            self.output_class = int(output_class)




