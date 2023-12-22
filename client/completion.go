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

type CompletionResponse struct {
	FinishReason string
	Content      string
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

type CompletionRequest struct {
	Model     ModelName
	Format    ResponseFormat
	MaxTokens int       // 0 means no limit, except for GPT4Vision, which has a small default limit
	Stream    io.Writer // if nil, then no streaming
	Messages  []openai.ChatCompletionMessage
}

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
		resp, err := c.CreateChatCompletion(context.Background(), req)
		if err != nil {
			return nil, err
		}
		buf, _ := json.MarshalIndent(resp, "", "  ")
		fmt.Println(string(buf))
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
		q := new(bytes.Buffer)
		out := io.MultiWriter(r.Stream, q)
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
			fmt.Fprint(out, firstChoice.Delta.Content)
		}
		fmt.Fprintln(out)
		return &CompletionResponse{
			FinishReason: finishReason,
			Content:      q.String(),
		}, nil
	}
}
