extends Node

var _port = "5000"
var _hostAddress = "http://127.0.0.1:" + _port

var url = _hostAddress + "/process_image_test"

var urlWithImage = _hostAddress + "/process_image"

var _urlWithImageName = _hostAddress + "/process_image_name/"
func urlWithImageNameVariable(name : String):
	return _urlWithImageName + name

var _urlWithFolderNameArchive = _hostAddress + "/archive/"
func urlWithFolderNameArchiveVariable(name : String):
	return _urlWithFolderNameArchive + name

var urlAllImageFileNames = _hostAddress + "/histological_images"
var urlAllArchiveFolderNames = _hostAddress + "/archive/folder"

var urlCurrentServerState = _hostAddress + "/state"

var httpResult : HTTPResult
var httpImageNames : HTTPImageNames

func create_instance(add : String):
	var scene = load(add)
	var scene_instance = scene.instantiate()
	return scene_instance
