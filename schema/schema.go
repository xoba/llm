package schema

import "github.com/invopop/jsonschema"

func Calculate(a any) *jsonschema.Schema {
	r := new(jsonschema.Reflector)
	r.ExpandedStruct = true
	return r.Reflect(a)
}
