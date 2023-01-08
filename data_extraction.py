import pickle

from comment import Comment
from collections import defaultdict

def tag_stripper(string):
    reading = False
    value = ""
    for letter in string:
        if letter == '>':
            reading = True
            continue
        if reading == True:
            if letter == "<":
                reading = False
            else:
                value = value + letter
    return value[:-1]

comment: Comment
comments = defaultdict(Comment)
index = 0

def read_and_save_xml_as_pickle(corpus):
    with open(f"{corpus}.xml", "r", encoding="utf-8") as data:
        for line in data:
            if line == "<comment>\n":
                comment = Comment()
            else:
                value = tag_stripper(line)
                if line[1:8] == "<movie>":
                    comment.set_movie(value)
                if line[1:12] == "<review_id>":
                    comment.set_review_id(value)
                if line[1:7] == "<name>":
                    comment.set_name(value)
                if line[1:10] == "<user_id>":
                    comment.set_user_id(value)
                if line[1:7] == "<note>":
                    comment.set_note(value)
                if line[1:14] == "<commentaire>":
                    comment.set_comment(value)
                if line == "</comment>\n":
                    comments[index] = comment
                    index+=1
    with open(f"{corpus}.pickle", "wb") as outfile:
        pickle.dump(comments, outfile)

read_and_save_xml_as_pickle("train")
read_and_save_xml_as_pickle("dev")
read_and_save_xml_as_pickle("test")