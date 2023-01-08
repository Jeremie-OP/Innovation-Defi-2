import json

from json import JSONEncoder

class MyEncoder(JSONEncoder):
    def default(self, obj):
        dict = {}
        dict["movie"] = obj.get_movie()
        dict["review_id"] = obj.get_review_id()
        dict["name"] = obj.get_name()
        dict["user_id"] = obj.get_user_id()
        dict["note"] = obj.get_note()
        dict["comment"] = obj.get_comment()
        return dict

class Comment():

    __movie: int
    __review_id: int
    __name: str
    __user_id: int
    __note: int
    __comment: str

    def set_movie(self, movie):
        self.__movie = movie

    def set_review_id(self, id):
        self.__review_id = id

    def set_name(self, name):
        self.__name = name

    def set_user_id(self, id):
        self.__user_id= id

    def set_note(self, note):
        self.__note = note

    def set_comment(self, comment):
        self.__comment = comment

    def get_movie(self):
        return self.__movie

    def get_review_id(self):
        return self.__review_id

    def get_name(self):
        return self.__name

    def get_user_id(self):
        return self.__user_id

    def get_note(self):
        return self.__note

    def get_comment(self):
        return self.__comment

    def __iter__(self):
        yield from {
            "movie": self.__movie,
            "review_id": self.__review_id,
            "name": self.__name,
            "user_id": self.__user_id,
            "note": self.__note,
            "comment": self.__comment
        }.items()

    def __str__(self):
        return json.dumps(dict(self), ensure_ascii=False)

    def __repr__(self):
        return self.__str__()