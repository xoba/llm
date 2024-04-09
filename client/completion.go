package client

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"regexp"
	"strconv"

	"github.com/sashabaranov/go-openai"
)

type CompletionRequest struct {
	Model     ModelName
	Format    ResponseFormat
	MaxTokens int       // 0 means no limit, except for GPT4Vision, which has a small default limit
	Stream    io.Writer // if nil, then no streaming
	Tools     []openai.Tool
	Messages  []openai.ChatCompletionMessage
}

type CompletionResponse struct {
	FinishReason  string
	Content       string
	FunctionCalls []*FunctionCall
}

//go:generate stringer -type=ResponseFormat
type ResponseFormat int

const (
	_ ResponseFormat = iota
	NoneSpecified
	JSONResponse
	TextResponse
)

//go:generate stringer -type=ModelName
type ModelName int

const (
	_ ModelName = iota
	DefaultModel
	GPT4Turbo
	GPT4Vision
)

func Complete(c OpenAI, r CompletionRequest) (*CompletionResponse, error) {
	var model string
	switch r.Model {
	case DefaultModel:
		model = "gpt-4-turbo"
	case GPT4Turbo, GPT4Vision:
		model = "gpt-4-turbo-2024-04-09"
	default:
		return nil, fmt.Errorf("unknown model: %d", r.Model)
	}
	req := openai.ChatCompletionRequest{
		Model:       model,
		Messages:    r.Messages,
		MaxTokens:   r.MaxTokens,
		Temperature: 1.0,
		TopP:        1,
		Tools:       r.Tools,
	}
	switch r.Format {
	case NoneSpecified:
	case JSONResponse:
		req.ResponseFormat = &openai.ChatCompletionResponseFormat{
			Type: openai.ChatCompletionResponseFormatTypeJSONObject,
		}
	case TextResponse:
		req.ResponseFormat = &openai.ChatCompletionResponseFormat{
			Type: openai.ChatCompletionResponseFormatTypeText,
		}
	default:
		return nil, fmt.Errorf("unknown format: %d", r.Format)
	}
	// TODO: unify stream-or-not processing, by aggregating
	// stream bits into an openai.ChatCompletionResponse,
	// not processing them specially:
	if r.Stream == nil {
		// TODO: need to handle functions here
		resp, err := c.CreateChatCompletion(context.Background(), req)
		if err != nil {
			return nil, err
		}
		return &CompletionResponse{
			FinishReason: string(resp.Choices[0].FinishReason),
			Content:      resp.Choices[0].Message.Content,
		}, nil
	} else {
		req.Stream = true
		resp, err := c.CreateChatCompletionStream(context.Background(), req)
		if err != nil {
			return nil, err
		}
		defer resp.Close()
		var calls []*FunctionCall
		content := new(bytes.Buffer)
		contentW := io.MultiWriter(r.Stream, content)
		var finishReason string
		for {
			t, err := resp.Recv()
			if err == io.EOF {
				break
			} else if err != nil {
				var apiError *openai.APIError
				if errors.As(err, &apiError) {
					p := regexp.MustCompile(`(\d+) tokens`)
					if p.MatchString(apiError.Message) {
						m := p.FindStringSubmatch(apiError.Message)
						tokens, convError := strconv.ParseUint(m[1], 10, 64)
						if convError != nil {
							return nil, convError
						}
						err = fmt.Errorf("total tokens = %d; error = %w", tokens, err)
					}
					return nil, err
				}
				return nil, err
			}
			if false {
				buf, err := json.MarshalIndent(t, "", "  ")
				if err != nil {
					return nil, err
				}
				fmt.Printf("\nresponse: %s\n", string(buf))
			}
			choices := t.Choices
			if len(choices) == 0 {
				return nil, fmt.Errorf("no choices")
			}
			firstChoice := choices[0]
			finishReason = string(firstChoice.FinishReason)
			fmt.Fprint(contentW, firstChoice.Delta.Content)
			if len(firstChoice.Delta.ToolCalls) > 0 {
				first := firstChoice.Delta.ToolCalls[0]
				if i := first.Index; i != nil {
					if *i >= len(calls) {
						calls = append(calls, &FunctionCall{
							w: new(bytes.Buffer),
						})
					}
					call := calls[*i]
					if len(first.ID) > 0 {
						call.ID = first.ID
					}
					if first.Type == "function" && len(first.Function.Name) > 0 {
						fmt.Printf("\nfunction: %s\nparameters: ", first.Function.Name)
						call.Name = first.Function.Name
					}
					call.Arguments += first.Function.Arguments
					fmt.Fprintf(r.Stream, first.Function.Arguments)
				}
			}
		}
		fmt.Fprintln(contentW)
		return &CompletionResponse{
			FinishReason:  finishReason,
			Content:       content.String(),
			FunctionCalls: calls,
		}, nil
	}
}

type FunctionCall struct {
	ID        string
	Name      string
	Arguments string
	w         *bytes.Buffer
}

func (f FunctionCall) String() string {
	buf, _ := json.MarshalIndent(f, "", "  ")
	return string(buf)
}
