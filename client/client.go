package client

import (
	"context"
	"io"
	"strings"

	"github.com/sashabaranov/go-openai"
)

type Interface interface {
	TranscribeAV(TranscriptionRequest) (string, error)
	Complete(CompletionRequest) (*CompletionResponse, error)
	OpenAI
}

type OpenAI interface {
	CreateChatCompletion(context.Context, openai.ChatCompletionRequest) (openai.ChatCompletionResponse, error)
	CreateChatCompletionStream(context.Context, openai.ChatCompletionRequest) (*openai.ChatCompletionStream, error)
	CreateTranscription(context.Context, openai.AudioRequest) (openai.AudioResponse, error)
	CreateImage(context.Context, openai.ImageRequest) (response openai.ImageResponse, err error)
	CreateTranslation(context.Context, openai.AudioRequest) (openai.AudioResponse, error)
	CreateEmbeddings(context.Context, openai.EmbeddingRequestConverter) (openai.EmbeddingResponse, error)
	CreateSpeech(context.Context, openai.CreateSpeechRequest) (io.ReadCloser, error)
}

type client struct {
	*openai.Client
}

func (c client) TranscribeAV(r TranscriptionRequest) (string, error) {
	return TranscribeAV(c, r)
}

func (c client) Complete(r CompletionRequest) (*CompletionResponse, error) {
	return Complete(c, r)
}

func New(key string) (Interface, error) {
	return client{openai.NewClient(strings.TrimSpace(key))}, nil
}
