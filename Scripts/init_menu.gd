extends CustomControl

class_name InitMenu

@onready var _startContainer : VBoxContainer = $Panel/StartContainer
@onready var _archiveContainer : VBoxContainer = $Panel/ArchiveContainer

@onready var _startButton : Button = $Panel/StartContainer/startButton
@onready var _refreshButton : Button = $Panel/StartContainer/refreshButton
@onready var _cancelButton : Button = $Panel/StartContainer/cancelButton

@onready var _loadArchiveButton : Button = $Panel/ArchiveContainer/loadArchiveButton
@onready var _refreshArchiveButton : Button = $Panel/ArchiveContainer/refreshArchiveButton
@onready var _cancelArchiveButton : Button = $Panel/ArchiveContainer/cancelArchiveButton

@onready var _imageNameList : ImageNameList = $Panel/StartContainer/ScrollContainer/ImageNamesList
@onready var _imageNameListArchive : ImageNameList = $Panel/ArchiveContainer/ScrollContainer/ImageNamesListArchive

var _currentMenu = "DEFAULT"

signal _on_initMenu_closed_with_start
signal _on_initMenu_closed_with_load

# Called when the node enters the scene tree for the first time.
func _ready():
	pass # Replace with function body.

func initalise():
	_startButton.connect("pressed", _start_button_pressed)
	_refreshButton.connect("pressed", _refresh_button_pressed)
	_cancelButton.connect("pressed", closeMenu)
	
	_loadArchiveButton.connect("pressed", _loadArchive_button_pressed)
	_refreshArchiveButton.connect("pressed", _refreshArchive_button_pressed)
	_cancelArchiveButton.connect("pressed", closeMenu)
	
	Global.httpImageNames.connect("image_file_names_loadet", _on_names_loadet)
	pass # Replace with function body.

func _on_names_loadet(names : Array):
	match _currentMenu:
		"start":
			_imageNameList.addItems(names)
		"archive":
			_imageNameListArchive.addItems(names)
	pass

# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta):
	pass

func showMenu(type : String):
	match type:
		"start":
			_startContainer.show()
			Global.httpImageNames.getAllImageFileNames()
		"archive":
			_archiveContainer.show()
			Global.httpImageNames.getAllArchiveFolderNames()
		_:
			return
	self.show()
	_currentMenu = type
	pass

func closeMenu():
	self.hide()
	_imageNameList.clear()
	_imageNameListArchive.clear()
	_startContainer.hide()
	_archiveContainer.hide()
	pass

func _start_button_pressed():
	Global.httpResult.getResultWithImageName(_imageNameList.getSelectedItemText())
	_on_initMenu_closed_with_start.emit()
	closeMenu()
	pass

func _refresh_button_pressed():
	_imageNameList.clear()
	Global.httpImageNames.getAllImageFileNames()
	pass

func _loadArchive_button_pressed():
	closeMenu()
	Global.httpResult.getArchiveWithFolderName(_imageNameListArchive.getSelectedItemText())
	_on_initMenu_closed_with_load.emit()
	pass

func _refreshArchive_button_pressed():
	_imageNameListArchive.clear()
	Global.httpResult.getAllArchiveFolderNames()
	pass
