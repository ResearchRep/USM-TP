%.dot.svg: %.dot
	dot -Tsvg $< -o$@

default: \
	task_graph.dot.svg \
	split_gemv.dot.svg \
	split_mlp.dot.svg
