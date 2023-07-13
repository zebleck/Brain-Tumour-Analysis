extends Control

class_name CustomControl

const _animationSpeed = 2

# Called when the node enters the scene tree for the first time.
func _ready():
	pass # Replace with function body.


# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta):
	pass

func move(target : Vector2):
	var tween = create_tween()
	tween.tween_property(self, "position", target - position, _animationSpeed).from(position).as_relative().set_trans(Tween.TRANS_QUINT).set_ease(Tween.EASE_OUT)
	pass

func move_with_callback(target : Vector2, callback : Callable):
	var tween = create_tween()
	tween.tween_property(self, "position", target - position, _animationSpeed).from(position).as_relative().set_trans(Tween.TRANS_QUINT).set_ease(Tween.EASE_OUT)
	tween.tween_callback(callback)
	pass
