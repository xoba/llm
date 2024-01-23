package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"strings"

	"github.com/sashabaranov/go-openai"
	"xoba.com/llm"
	"xoba.com/llm/client"
	"xoba.com/llm/schema"
)

func main() {
	if err := router("conversation"); err != nil {
		fmt.Fprintln(os.Stderr, err)
	}
}

func router(mode string) error {
	if len(os.Args) > 1 {
		mode = os.Args[1]
	}
	var f func(client.Interface) error
	switch mode {
	case "arithmetic":
		f = arithmetic
	case "conversation":
		f = conversation
	default:
		return fmt.Errorf("unknown mode: %q", mode)
	}
	c, err := client.NewDefault()
	if err != nil {
		return err
	}
	return f(c)
}

type ConversationResponse struct {
	StoryTitle string
	Actors     []string
}

func conversation(c client.Interface) error {
	var messages []openai.ChatCompletionMessage
	prompt := "create a short story concept for me"
	reader := bufio.NewReader(os.Stdin)
	for {
		r, err := llm.Ask(c, llm.Question[ConversationResponse]{
			Prompt:   prompt,
			Messages: messages,
		})
		if err != nil {
			return err
		}
		fmt.Print("\n> ")
		text, err := reader.ReadString('\n')
		if err == io.EOF {
			fmt.Println()
			return nil
		} else if err != nil {
			return fmt.Errorf("can't read from stdin: %v", err)
		}
		prompt = strings.TrimSpace(text)
		messages = r.Messages
	}
}

func arithmetic(c client.Interface) error {
	tools := make(map[string]llm.Tool)
	for _, t := range []llm.Tool{Sum{}, Mult{}, Exp{}} {
		tools[t.Defintion().Name] = t
	}
	const question = "what is 5 * (455342+22342.6)^1.1 * 99?"
	fmt.Printf("question: %q\n", question)
	r, err := llm.Ask(c, llm.Question[ArithmeticResponse]{
		Prompt: question,
		Tools:  tools,
		Examples: []llm.Example[ArithmeticResponse]{
			{
				Prompt: "what is 7 + 3?",
				Answer: ArithmeticResponse{
					Answer:           "10",
					DifficultyRating: "easy",
				},
			},
			{
				Prompt: "what is (10*2)^2?",
				Answer: ArithmeticResponse{
					Answer:           "400",
					DifficultyRating: "medium",
				},
			},
		},
	})
	if err != nil {
		return err
	}
	fmt.Printf("conversational: %q\n", r.Answer.ConversationalAnswer)
	fmt.Printf("formal: %q\n", r.Answer.FormalAnswer.Answer)
	fmt.Printf("difficulty: %q\n", r.Answer.FormalAnswer.DifficultyRating)
	for _, m := range r.Messages {
		buf, err := json.MarshalIndent(m, "", "  ")
		if err != nil {
			return err
		}
		fmt.Println(string(buf))
	}
	return nil
}

type ArithmeticResponse struct {
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
