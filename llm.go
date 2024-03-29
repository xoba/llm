// package llmif is an interface to llm's via json structs
package llm

import (
	"bytes"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"os/exec"
	"sort"
	"strings"

	"github.com/invopop/jsonschema"
	"github.com/sashabaranov/go-openai"
	"github.com/vincent-petithory/dataurl"
	"xoba.com/llm/assets"
	"xoba.com/llm/client"
	"xoba.com/llm/schema"
)

// Question is a question to ask the assistant
// ANSWER is the type of the answer, field names should be self-explanatory
type Question[ANSWER any] struct {
	Prompt   string                         // the question to ask, including prompt etc.
	Files    []File                         // files to use as background material (additional each round)
	Examples []Example[ANSWER]              // examples of what the answer may look like (additional each round)
	Tools    map[string]Tool                // tools at the assistant's disposal
	Messages []openai.ChatCompletionMessage // state of prior conversation
}

type Example[ANSWER any] struct {
	Prompt string
	Answer ANSWER
}

// File is background material
type File struct {
	Name        string
	Content     []byte
	ContentType string
}

type Tool interface {
	Defintion() openai.FunctionDefinition
	Compute(parameters string) (string, error)
}

// Answer is the response to asking a Question
type Answer[ANSWER any] struct {
	ConversationalAnswer string // free-form, high-level answer to the question
	FormalAnswer         ANSWER // formal answer to the question, in a specific schema
}

type Response[ANSWER any] struct {
	Answer   *Answer[ANSWER]
	Messages []openai.ChatCompletionMessage
}

func (r Answer[T]) String() string {
	buf, _ := json.MarshalIndent(r, "", "  ")
	return string(buf)
}

func Ask[ANSWER any](c client.Interface, q Question[ANSWER]) (*Response[ANSWER], error) {
	firstQuestion := len(q.Messages) == 0
	add := func(m openai.ChatCompletionMessage) {
		q.Messages = append(q.Messages, m)
	}
	if firstQuestion {
		add(openai.ChatCompletionMessage{
			Role:    openai.ChatMessageRoleSystem,
			Content: assets.Prompt1,
		})
		add(openai.ChatCompletionMessage{
			Role:    openai.ChatMessageRoleSystem,
			Content: assets.Prompt2,
		})
	}
	whichModel := client.GPT4Turbo
	responseFormat := client.JSONResponse
	var maxTokens int
	if len(q.Files) > 0 {
		add(openai.ChatCompletionMessage{
			Role:    openai.ChatMessageRoleSystem,
			Content: fmt.Sprintf(`there are going to be %d files in the following request, each of which you will use as background material for assisting the user.`, len(q.Files)),
		})
	}
	for _, d := range q.Files {
		switch d.ContentType {
		case "audio/mp3", "audio/mp4", "audio/mpeg", "audio/wav", "audio/x-wav", "audio/webm", "video/mp4", "video/mpeg", "video/webm":
			txt, err := c.TranscribeAV(client.TranscriptionRequest{
				File: client.AVFile{ContentType: d.ContentType, Content: d.Content},
			})
			if err != nil {
				return nil, err
			}
			add(openai.ChatCompletionMessage{
				Role: openai.ChatMessageRoleSystem,
				Content: fmt.Sprintf(
					"here is the transcription of a %s file named %q:\n\n%s",
					d.ContentType, d.Name,
					txt,
				),
			})
		case "application/pdf":
			txt, err := pdfToText(d.Content)
			if err != nil {
				return nil, err
			}
			add(openai.ChatCompletionMessage{
				Role: openai.ChatMessageRoleSystem,
				Content: fmt.Sprintf(
					"here is the text rendering of an %s file named %q:\n\n%s",
					d.ContentType, d.Name,
					string(txt),
				),
			})
		case "application/json",
			"text/plain", "text/html", "text/markdown", "text/csv", "text/xml", "text/rtf",
			"text/tab-separated-values", "text/richtext",
			"text/yaml", "text/x-yaml", "text/x-markdown", "text/x-rst", "text/x-org":
			add(openai.ChatCompletionMessage{
				Role: openai.ChatMessageRoleSystem,
				Content: fmt.Sprintf(
					"here is a %s file named %q:\n\n%s",
					d.ContentType, d.Name,
					string(d.Content),
				),
			})
		case "image/png", "image/jpeg", "image/webp", "image/gif":
			whichModel = client.GPT4Vision
			responseFormat = client.NoneSpecified // vision has no format at all
			maxTokens = 4096                      // specify, since vision defaults to few output tokens
			add(openai.ChatCompletionMessage{
				Role: openai.ChatMessageRoleSystem,
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
			return nil, fmt.Errorf("unsupported content type: %q", d.ContentType)
		}
	}
	add(openai.ChatCompletionMessage{
		Role:    openai.ChatMessageRoleUser,
		Content: q.Prompt,
	})
	if firstQuestion {
		responseSchema := schema.Calculate(&Answer[ANSWER]{})
		schema, err := json.MarshalIndent(responseSchema, "", "  ")
		if err != nil {
			return nil, err
		}
		add(openai.ChatCompletionMessage{
			Role:    openai.ChatMessageRoleUser,
			Content: fmt.Sprintf(`the schema of your json answer must match: %s`, string(schema)),
		})
	}
	if len(q.Examples) > 0 {
		examples := new(bytes.Buffer)
		fmt.Fprintf(examples, "here are %d fictitious example(s) for how your json responses may look like in practice:\n\n", len(q.Examples))
		for i, e := range q.Examples {
			a := Answer[ANSWER]{
				FormalAnswer:         e.Answer,
				ConversationalAnswer: cleanText(fmt.Sprintf(`freeform text about answering question %q`, e.Prompt)),
			}
			buf, err := json.MarshalIndent(a, "", "  ")
			if err != nil {
				return nil, err
			}
			fmt.Fprintf(examples, "example #%d in response to prompt %q: %s\n\n", i+1, e.Prompt, string(buf))
		}
		add(openai.ChatCompletionMessage{
			Role:    openai.ChatMessageRoleUser,
			Content: examples.String(),
		})
	}
	var errs []error
	var tools []openai.Tool
	switch whichModel {
	case client.GPT4Turbo:
		for name, t := range q.Tools {
			def := t.Defintion()
			if name != def.Name {
				return nil, fmt.Errorf("tool name %q does not match definition name %q", name, def.Name)
			}
			tools = append(tools, openai.Tool{
				Type:     openai.ToolTypeFunction,
				Function: &def,
			})
		}
		sort.Slice(tools, func(i, j int) bool {
			return tools[i].Function.Name < tools[j].Function.Name
		})
	case client.GPT4Vision:
		// tools not supported by GPTV!
	}
LOOP:
	for {
		if len(errs) > 4 {
			return nil, fmt.Errorf("too many tries: %v", errs)
		}
		resp, err := c.Complete(client.CompletionRequest{
			Model:     whichModel,
			Format:    responseFormat,
			MaxTokens: maxTokens,
			Stream:    os.Stdout,
			Messages:  q.Messages,
			Tools:     tools,
		})
		if err != nil {
			return nil, err
		}
		switch resp.FinishReason {
		case "tool_calls":
			for _, call := range resp.FunctionCalls {
				tool, ok := q.Tools[call.Name]
				if !ok {
					return nil, fmt.Errorf("unknown tool: %q", call.Name)
				}
				result, err := tool.Compute(call.Arguments)
				if err != nil {
					return nil, err
				}
				fmt.Printf("result = %s\n", result)
				add(openai.ChatCompletionMessage{
					Role: openai.ChatMessageRoleAssistant,
					ToolCalls: []openai.ToolCall{
						{
							ID:   call.ID,
							Type: openai.ToolTypeFunction,
							Function: openai.FunctionCall{
								Name:      call.Name,
								Arguments: call.Arguments,
							},
						},
					},
				})
				add(openai.ChatCompletionMessage{
					Role:       openai.ChatMessageRoleTool,
					Content:    result,
					ToolCallID: call.ID,
				})
			}
			continue LOOP

		case "stop":
			add(openai.ChatCompletionMessage{
				Role:    openai.ChatMessageRoleAssistant,
				Content: resp.Content,
			})
			// minor cleanup if needed:
			resp.Content = strings.TrimSpace(resp.Content)
			resp.Content = strings.TrimPrefix(resp.Content, "```json")
			resp.Content = strings.TrimSuffix(resp.Content, "```")
			d := json.NewDecoder(strings.NewReader(resp.Content))
			d.DisallowUnknownFields()
			var parsedResponse Answer[ANSWER]
			if err := d.Decode(&parsedResponse); err != nil {
				log.Printf("error decoding, going to potentially retry: %v", err)
				errs = append(errs, err)
				add(openai.ChatCompletionMessage{
					Role:    openai.ChatMessageRoleUser,
					Content: fmt.Sprintf("oops, i got an error parsing your response as json: %v. could you please re-do with correct json having no extraneous characters, etc?", err),
				})
				continue LOOP
			}
			return &Response[ANSWER]{
				Answer:   &parsedResponse,
				Messages: q.Messages,
			}, nil
		default:
			return nil, fmt.Errorf("unhandled finish reason: %q", resp.FinishReason)
		}
	}
}

func pdfToText(pdf []byte) ([]byte, error) {
	cmd := exec.Command("pdftotext", "-", "-")
	cmd.Stdin = bytes.NewReader(pdf)
	return cmd.Output()
}

func cleanText(s string) string {
	return strings.Join(strings.Fields(s), " ")
}

// BOGUS: jsonschema MarshalJSON doesn't work unless we import it HERE, via this decl!!!
var _ jsonschema.Reflector
