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
	Answer           string
	DifficultyRating string
}

type Sum struct {
	Addends []float64
}

func (s Sum) Defintion() openai.FunctionDefinition {
	return openai.FunctionDefinition{
		Name:        "sum",
		Description: "adds numbers",
		Parameters:  schema.Calculate(s),
	}
}

func (s Sum) Compute(p string) (string, error) {
	if err := json.Unmarshal([]byte(p), &s); err != nil {
		return "", err
	}
	var sum float64
	for _, x := range s.Addends {
		sum += x
	}
	return fmt.Sprintf("%f", sum), nil
}

type Mult struct {
	Multiplicands []float64
}

func (s Mult) Defintion() openai.FunctionDefinition {
	return openai.FunctionDefinition{
		Name:        "mult",
		Description: "multiplies numbers",
		Parameters:  schema.Calculate(s),
	}
}

func (s Mult) Compute(p string) (string, error) {
	if err := json.Unmarshal([]byte(p), &s); err != nil {
		return "", err
	}
	product := 1.0
	for _, x := range s.Multiplicands {
		product *= x
	}
	return fmt.Sprintf("%f", product), nil
}

func run() error {
	c, err := client.NewDefault()
	if err != nil {
		return err
	}
	tools := make(map[string]llm.Tool)
	for _, t := range []llm.Tool{Sum{}, Mult{}} {
		tools[t.Defintion().Name] = t
	}
	const question = "what is 5*(455342+22342.6)*99?"
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
	fmt.Printf("difficulty: %q\n", r.Answer.DifficultyRating)
	return nil
}
