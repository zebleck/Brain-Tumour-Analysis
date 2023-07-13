extends HTTPRequestCustom

class_name HTTPImageNames

var _header = ["Content-Type: application/json; charset=UTF-8"]

signal image_file_names_loadet(imageNames : Array)
signal request_failed(error : String, info : String )
signal current_server_state(state : String)

func getAllImageFileNames():
	self.add_to_queue(Global.urlAllImageFileNames, _header, HTTPClient.METHOD_GET, "", _handleResultAllImageFileNames)
	pass

func getAllArchiveFolderNames():
	self.add_to_queue(Global.urlAllArchiveFolderNames, _header, HTTPClient.METHOD_GET, "", _handleResultAllImageFileNames)
	pass

func _handleResultAllImageFileNames(result, response_code, header, body):
	if(response_code == 200):
		var json = JSON.parse_string(body.get_string_from_utf8())
		image_file_names_loadet.emit(json.images)
	else:
		var error = "Error - HTTPImageNames (Folder too): " + str(response_code)
		var info = JSON.parse_string(body.get_string_from_utf8())
		print(error)
		print(info)
		request_failed.emit(error, "info")
		pass
	pass

func getProcessUpdate():
	self.add_to_queue(Global.urlCurrentServerState, _header, HTTPClient.METHOD_GET, "", _handleProcessUpdate)
	pass

func _handleProcessUpdate(result, response_code, header, body):
	if(response_code == 200):
		var json = JSON.parse_string(body.get_string_from_utf8())
		current_server_state.emit(json.state)
	else:
		var error = "Error - HTTPImageNames Process Update: " + str(response_code)
		var info = JSON.parse_string(body.get_string_from_utf8())
		print(error)
		print(info)
		request_failed.emit(error, "info")
		pass
	pass
