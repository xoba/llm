package main

import (
	"fmt"

	"xoba.com/llm"
	"xoba.com/llm/client"
)

func main() {
	if err := run(); err != nil {
		fmt.Println(err)
	}
}

type Response struct {
	Answer string
}

func run() error {
	c, err := client.NewDefault()
	if err != nil {
		return err
	}
	r, err := llm.Ask[Response](c, llm.Question[Response]{
		Prompt: "what is 2+2?",
	})
	if err != nil {
		return err
	}
	fmt.Printf("meta: %q\n", r.Meta)
	fmt.Printf("answer: %q\n", r.Answer.Answer)
	return nil
}
