package client

import (
	"bytes"
	"context"
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
	FinishReason string
	Content      string
	FunctionCall *FunctionCall
}

//go:generate stringer -type=ResponseFormat
type ResponseFormat int

const (
	_             ResponseFormat = iota
	NoneSpecified                // vision preview needs this
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
		model = openai.GPT4TurboPreview
	case GPT4Turbo:
		model = openai.GPT4TurboPreview
	case GPT4Vision:
		model = openai.GPT4VisionPreview
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
		content := new(bytes.Buffer)
		funcs := new(bytes.Buffer)
		var funcName, callID string
		contentW := io.MultiWriter(r.Stream, content)
		funcsW := io.MultiWriter(r.Stream, funcs)
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
			choices := t.Choices
			if len(choices) == 0 {
				return nil, fmt.Errorf("no choices")
			}
			firstChoice := choices[0]
			finishReason = string(firstChoice.FinishReason)
			fmt.Fprint(contentW, firstChoice.Delta.Content)
			if len(firstChoice.Delta.ToolCalls) > 0 {
				first := firstChoice.Delta.ToolCalls[0]
				if len(first.ID) > 0 {
					callID = first.ID
				}
				if first.Type == "function" && len(first.Function.Name) > 0 {
					fmt.Printf("function: %s\nparameters: ", first.Function.Name)
					funcName = first.Function.Name
				}
				fmt.Fprintf(funcsW, first.Function.Arguments)
			}
		}
		fmt.Fprintln(contentW)
		var fc *FunctionCall
		if len(funcName) > 0 {
			fc = &FunctionCall{
				ID:        callID,
				Name:      funcName,
				Arguments: funcs.String(),
			}
		}
		return &CompletionResponse{
			FinishReason: finishReason,
			Content:      content.String(),
			FunctionCall: fc,
		}, nil
	}
}

type FunctionCall struct {
	ID        string
	Name      string
	Arguments string
}
