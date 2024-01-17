package main

import (
	"encoding/json"
	"fmt"

	"github.com/sashabaranov/go-openai"
	"xoba.com/llm"
	"xoba.com/llm/client"
	"xoba.com/llm/schema"
)

func main() {
	if err := run(); err != nil {
		fmt.Println(err)
	}
}

type Response struct {
	Answer string
}

type Sum struct {
	A, B float64
}

func (s Sum) Defintion() openai.FunctionDefinition {
	return openai.FunctionDefinition{
		Name:        "sum",
		Description: "adds two numbers",
		Parameters:  schema.Calculate(s),
	}
}

func (s Sum) Compute(p string) (string, error) {
	if err := json.Unmarshal([]byte(p), &s); err != nil {
		return "", err
	}
	return fmt.Sprintf("%f", s.A+s.B), nil
}

func run() error {
	c, err := client.NewDefault()
	if err != nil {
		return err
	}
	tools := make(map[string]llm.Tool)
	for _, t := range []llm.Tool{Sum{}} {
		tools[t.Defintion().Name] = t
	}
	// const question = "what is most likely color of an apple? one word answer please"
	const question = "what is 455342+22342.6?"
	fmt.Printf("question: %q\n", question)
	r, err := llm.Ask[Response](c, llm.Question[Response]{
		Prompt: question,
		Tools:  tools,
	})
	if err != nil {
		return err
	}
	fmt.Printf("meta: %q\n", r.Meta)
	fmt.Printf("answer: %q\n", r.Answer.Answer)
	return nil
}
