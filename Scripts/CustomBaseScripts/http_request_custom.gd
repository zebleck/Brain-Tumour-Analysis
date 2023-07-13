extends HTTPRequest

class_name HTTPRequestCustom

var queue = []
var running : bool = false
var currentRequestData : RequestData

func _init():
	self.connect("request_completed", _request_completed)
	self.use_threads = true
	pass
	
func _process(delta):
	if(not running and not queue.is_empty()):
		_callQueue()
		pass
	pass

# Add an item to the queue if it has not reached its maximum capacity
func add_to_queue(path : String, header : PackedStringArray, method : int, content : String, handleResponse : Callable):
	print("add_to_queque: " + path)
	queue.push_back(RequestData.new(path, header, method, content, handleResponse))
	pass
	
func _callQueue():
	currentRequestData = queue.pop_front()
	if(currentRequestData != null):
		running = true
		_callApi()
		pass
	pass

func _callApi():
	self.connect("request_completed", currentRequestData.handleResponse)
	print("_callApi: " + currentRequestData.path)
	self.request(currentRequestData.path, currentRequestData.header, currentRequestData.method, currentRequestData.content)
	pass

func _request_completed(result, response_code, header, body):
	self.disconnect("request_completed", currentRequestData.handleResponse)
	running = false
	pass
