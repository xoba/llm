package assets

import (
	"embed"
	"io/fs"
)

//go:embed *.txt
var vfs embed.FS

var Prompt string

func init() {
	buf, err := fs.ReadFile(vfs, "prompt.txt")
	if err != nil {
		panic(err)
	}
	Prompt = string(buf)
}
