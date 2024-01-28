package assets

import (
	"embed"
	"io/fs"
)

//go:embed *.txt
var vfs embed.FS

var Prompt1, Prompt2 string

func load(name string) (string, error) {
	buf, err := fs.ReadFile(vfs, name)
	if err != nil {
		return "", err
	}
	return string(buf), nil
}

func init() {
	p, err := load("prompt1.txt")
	if err != nil {
		panic(err)
	}
	Prompt1 = p
}

func init() {
	p, err := load("prompt2.txt")
	if err != nil {
		panic(err)
	}
	Prompt2 = p
}
