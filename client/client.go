package client

import (
	"context"
	"fmt"
	"os"
	"strings"

	"github.com/sashabaranov/go-openai"
)

type Client interface {
	TranscribeAV(AVFile) (string, error)
	Complete(CompletionRequest) (*CompletionResponse, error)
	OpenAIClient
}

type OpenAIClient interface {
	CreateChatCompletion(context.Context, openai.ChatCompletionRequest) (openai.ChatCompletionResponse, error)
	CreateChatCompletionStream(context.Context, openai.ChatCompletionRequest) (*openai.ChatCompletionStream, error)
	CreateTranscription(context.Context, openai.AudioRequest) (openai.AudioResponse, error)
	CreateTranslation(context.Context, openai.AudioRequest) (openai.AudioResponse, error)
	CreateImage(context.Context, openai.ImageRequest) (response openai.ImageResponse, err error)
}

type client struct {
	*openai.Client
}

func (c client) TranscribeAV(f AVFile) (string, error) {
	return TranscribeAV(c, f)
}

func (c client) Complete(r CompletionRequest) (*CompletionResponse, error) {
	return Complete(c, r)
}

func NewClient(key string) (Client, error) {
	return client{openai.NewClient(strings.TrimSpace(key))}, nil
}

// key via openai env var, or file openai.txt
func NewDefaultClient() (Client, error) {
	key, err := loadKey()
	if err != nil {
		return nil, err
	}
	return NewClient(key)
}

func loadKey() (string, error) {
	const (
		env  = "openai"
		file = "openai.txt"
	)
	key := os.Getenv(env)
	if len(key) == 0 {
		x, err := LoadFile(file)
		if err != nil {
			return "", err
		}
		key = x
	}
	const prefix = "sk-"
	if !strings.HasPrefix(key, prefix) {
		return "", fmt.Errorf("openai key should start with %q prefix", prefix)
	}
	return key, nil
}

func LoadFile(file string) (string, error) {
	buf, err := os.ReadFile(file)
	if err != nil {
		return "", err
	}
	return strings.TrimSpace(string(buf)), nil
}