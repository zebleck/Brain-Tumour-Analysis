extends Node

class_name RequestData

var path : String
var header : PackedStringArray
var method : int
var content : String
var handleResponse : Callable

func _init(path : String, header : PackedStringArray, method : int, content : String, handleResponse : Callable):
	self.path = path
	self.header = header
	self.method = method
	self.content = content
	self.handleResponse = handleResponse
	pass

