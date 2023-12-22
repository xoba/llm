// package llmif is an interface to llm's via json structs
package llm

import (
	"bytes"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"os/exec"
	"strings"

	"github.com/invopop/jsonschema"
	"github.com/sashabaranov/go-openai"
	"github.com/vincent-petithory/dataurl"
	"xoba.com/llm/client"
)

type Question[ANSWER any] struct {
	Prompt   string
	Files    []File
	Examples []ANSWER
}

type File struct {
	Name        string
	Content     []byte
	ContentType string
}

type Response[T any] struct {
	Meta   string // free-form meta information about the process
	Answer T
}

func (r Response[T]) String() string {
	buf, _ := json.MarshalIndent(r, "", "  ")
	return string(buf)
}

func Ask[ANSWER any](q Question[ANSWER]) (Response[ANSWER], error) {
	var zero Response[ANSWER]
	c, err := client.NewDefaultClient()
	if err != nil {
		return zero, err
	}
	var messages []openai.ChatCompletionMessage
	messages = append(messages, openai.ChatCompletionMessage{
		Role:    "system",
		Content: "you respond only in JSON, please, without enclosing markdown, or any other extraneous narrative or characters. this is very important for my career!",
	})
	whichModel := client.GPT4Turbo
	responseFormat := client.JSONResponse
	var maxTokens int
	messages = append(messages, openai.ChatCompletionMessage{
		Role:    "system",
		Content: fmt.Sprintf(`there are going to be %d files in the following request, each of which you will use as background material for assisting the user.`, len(q.Files)),
	})
	for _, d := range q.Files {
		switch d.ContentType {
		case "audio/mp3", "audio/mp4", "audio/mpeg", "audio/wav", "audio/x-wav", "audio/webm", "video/mp4", "video/mpeg", "video/webm":
			txt, err := c.TranscribeAV(client.AVFile{
				ContentType: d.ContentType,
				Content:     d.Content,
			})
			if err != nil {
				return zero, err
			}
			messages = append(messages, openai.ChatCompletionMessage{
				Role: "system",
				Content: fmt.Sprintf(
					"here is the transcription of a %s file named %q:\n\n%s",
					d.ContentType, d.Name,
					txt,
				),
			})
		case "application/pdf":
			txt, err := PdfToText(d.Content)
			if err != nil {
				return zero, err
			}
			messages = append(messages, openai.ChatCompletionMessage{
				Role: "system",
				Content: fmt.Sprintf(
					"here is the text rendering of an %s file named %q:\n\n%s",
					d.ContentType, d.Name,
					string(txt),
				),
			})
		case "text/plain", "text/html", "text/markdown", "text/csv", "text/tab-separated-values", "text/rtf", "text/richtext", "text/xml", "text/yaml", "text/x-yaml", "text/x-markdown", "text/x-rst", "text/x-org":
			messages = append(messages, openai.ChatCompletionMessage{
				Role: "system",
				Content: fmt.Sprintf(
					"here is a %s file named %q:\n\n%s",
					d.ContentType, d.Name,
					string(d.Content),
				),
			})
		case "image/png", "image/jpeg", "image/webp", "image/gif":
			whichModel = client.GPT4Vision
			responseFormat = client.NoneSpecified // vision has no format at all
			maxTokens = 1000                      // vision defaults to few output tokens
			messages = append(messages, openai.ChatCompletionMessage{
				Role: "system",
				MultiContent: []openai.ChatMessagePart{
					{
						Type: openai.ChatMessagePartTypeText,
						Text: fmt.Sprintf(
							"here is an %s file named %q",
							d.ContentType, d.Name,
						),
					},
					{
						Type: openai.ChatMessagePartTypeImageURL,
						ImageURL: &openai.ChatMessageImageURL{
							URL: dataurl.New(d.Content, d.ContentType).String(),
						},
					},
				},
			})
		default:
			return zero, fmt.Errorf("unsupported content type: %q", d.ContentType)
		}
	}
	messages = append(messages, openai.ChatCompletionMessage{
		Role:    "user",
		Content: q.Prompt,
	})
	{
		responseSchema := schema(&Response[ANSWER]{})
		schema, err := json.MarshalIndent(responseSchema, "", "  ")
		if err != nil {
			return zero, err
		}
		messages = append(messages, openai.ChatCompletionMessage{
			Role:    "user",
			Content: fmt.Sprintf(`the schema of your json answer must match: %s`, string(schema)),
		})
	}
	if len(q.Examples) > 0 {
		examples := new(bytes.Buffer)
		fmt.Fprintf(examples, "here are %d fictitious example(s) for how your json response may look like in practice:\n\n", len(q.Examples))
		for i, e := range q.Examples {
			a := Response[ANSWER]{
				Answer: e,
				Meta: cleanText(`this is a free-form response field for meta-level information, 
			it can be anything related to the task at hand or 
			the conversational process, not necessarily the question or answer per se.`),
			}
			buf, err := json.MarshalIndent(a, "", "  ")
			if err != nil {
				return zero, err
			}
			fmt.Fprintf(examples, "example #%d: %s\n\n", i+1, string(buf))
		}
		messages = append(messages, openai.ChatCompletionMessage{
			Role:    "user",
			Content: examples.String(),
		})
	}
	var errs []error
	var n int
	for {
		if n++; n == 4 {
			return zero, fmt.Errorf("too many tries: %v", errs)
		}
		resp, err := c.Complete(client.CompletionRequest{
			Model:     whichModel,
			Format:    responseFormat,
			MaxTokens: maxTokens,
			Stream:    os.Stdout,
			Messages:  messages,
		})
		if err != nil {
			return zero, err
		}
		messages = append(messages, openai.ChatCompletionMessage{
			Role:    "assistant",
			Content: resp.Content,
		})
		// minor cleanup if needed:
		resp.Content = strings.TrimSpace(resp.Content)
		resp.Content = strings.TrimPrefix(resp.Content, "```json")
		resp.Content = strings.TrimSuffix(resp.Content, "```")
		d := json.NewDecoder(strings.NewReader(resp.Content))
		d.DisallowUnknownFields()
		var parsedResponse Response[ANSWER]
		if err := d.Decode(&parsedResponse); err != nil {
			log.Printf("error decoding: %v", err)
			errs = append(errs, err)
			messages = append(messages, openai.ChatCompletionMessage{
				Role:    "user",
				Content: fmt.Sprintf("oops, i got an error parsing your response as json: %v. could you please re-do with correct json having no extraneous characters, etc?", err),
			})
			continue
		}
		return parsedResponse, nil
	}
}

func PdfToText(pdf []byte) ([]byte, error) {
	cmd := exec.Command("pdftotext", "-", "-")
	cmd.Stdin = bytes.NewReader(pdf)
	return cmd.Output()
}

func cleanText(s string) string {
	return strings.Join(strings.Fields(s), " ")
}

func schema(a any) any {
	r := new(jsonschema.Reflector)
	r.ExpandedStruct = true
	return r.Reflect(a)
}
