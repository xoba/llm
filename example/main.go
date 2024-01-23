package main

import (
	"encoding/json"
	"fmt"
	"math"
	"os"

	"github.com/sashabaranov/go-openai"
	"xoba.com/llm"
	"xoba.com/llm/client"
	"xoba.com/llm/schema"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
	}
}

func run() error {
	c, err := client.NewDefault()
	if err != nil {
		return err
	}
	tools := make(map[string]llm.Tool)
	for _, t := range []llm.Tool{Sum{}, Mult{}, Exp{}} {
		tools[t.Defintion().Name] = t
	}
	const question = "what is 5 * (455342+22342.6)^1.1 * 99?"
	fmt.Printf("question: %q\n", question)
	r, err := llm.Ask[Response](c, llm.Question[Response]{
		Prompt: question,
		Tools:  tools,
		Examples: []llm.Example[Response]{
			{
				Prompt: "what is 7 + 3?",
				Answer: Response{
					Answer:           "10",
					DifficultyRating: "easy",
				},
			},
			{
				Prompt: "what is (10*2)^2?",
				Answer: Response{
					Answer:           "400",
					DifficultyRating: "medium",
				},
			},
		},
	})
	if err != nil {
		return err
	}
	fmt.Printf("meta: %q\n", r.Response.Meta)
	fmt.Printf("answer: %q\n", r.Response.Answer.Answer)
	fmt.Printf("difficulty: %q\n", r.Response.Answer.DifficultyRating)
	for _, m := range r.Messages {
		buf, err := json.MarshalIndent(m, "", "  ")
		if err != nil {
			return err
		}
		fmt.Println(string(buf))
	}
	return nil
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

type Exp struct {
	Base  float64
	Power float64
}

func (s Exp) Defintion() openai.FunctionDefinition {
	return openai.FunctionDefinition{
		Name:        "exp",
		Description: "exponentiation",
		Parameters:  schema.Calculate(s),
	}
}

func (s Exp) Compute(p string) (string, error) {
	if err := json.Unmarshal([]byte(p), &s); err != nil {
		return "", err
	}
	return fmt.Sprintf("%f", math.Pow(s.Base, s.Power)), nil
}
