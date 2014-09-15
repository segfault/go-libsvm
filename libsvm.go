// Simple LIBSVM wrapper for Go.
package libsvm

/*
#cgo LDFLAGS: -lsvm -lm
#include <svm.h>
#include <stdlib.h>

const struct svm_node TERMINATOR = (struct svm_node) { -1, 0.0 };

static void model_free(struct svm_model *model) {
	svm_free_and_destroy_model(&model);
}
*/
import "C"

import (
	"fmt"
	"unsafe"
)

type SvmType int
type KernelType int

const (
	C_SVC       = SvmType(C.C_SVC)
	NU_SVC      = SvmType(C.NU_SVC)
	ONE_CLASS   = SvmType(C.ONE_CLASS)
	EPSILON_SVR = SvmType(C.EPSILON_SVR)
	NU_SVR      = SvmType(C.NU_SVR)

	LINEAR      = KernelType(C.LINEAR)
	POLY        = KernelType(C.POLY)
	RBF         = KernelType(C.RBF)
	SIGMOID     = KernelType(C.SIGMOID)
	PRECOMPUTED = KernelType(C.PRECOMPUTED)
)

// SvmError wraps LIBSVM failures so they can be handled
type SvmError struct {
	Message string
}

// SvmProblem is a wrapper around the svm_problem struct
type SvmProblem struct {
	object *C.struct_svm_problem
}

// SvmParameter is a wrapper around the svm_parameter struct
type SvmParameter struct {
	object *C.struct_svm_parameter
}

// SvmModel is a wrapper around the svm_model struct.
// The intent here is to provide convenience functions in a go-like way
type SvmModel struct {
	object *C.struct_svm_model
}

// SvmNode is a wrapper around the svm_node struct.
type SvmNode struct {
	object *C.struct_svm_node
	length int
}

// Version will return the libsvm version
func Version() int {
	return int(C.libsvm_version)
}

func NewExample(startIndex int, data []float64) *SvmNode {
	sidx := startIndex
	res := make([]C.struct_svm_node, len(data)+1)

	for i, v := range data {
		res[i].index = C.int(sidx + i)
		res[i].value = C.double(v)
	}

	res[len(data)].index = -1
	res[len(data)].value = 0

	return &SvmNode{
		object: &res[0],
		length: len(data),
	}
}

// Free will free memory allocated to the node's internal svm_node object(s)
func (node *SvmNode) Free() {
	C.free(unsafe.Pointer(node.object))
	node.length = 0
	node.object = nil
}

// Train a model for the given problem using the provided parameters.
// Will return a model or an error
func Train(prob SvmProblem, param SvmParameter) (*SvmModel, error) {
	mdl := C.svm_train(prob.object, param.object)
	if mdl == nil {
		return nil, SvmError{Message: "error while training. nil model returned"}
	}

	return &SvmModel{object: mdl}, nil
}

// Load a model from disk. This will return an error message if
// there is a problem loading from disk.
func Load(filename string) (*SvmModel, error) {

	cfn := C.CString(filename)
	defer C.free(unsafe.Pointer(cfn))

	mdl := C.svm_load_model(cfn)
	if mdl == nil {
		return nil, SvmError{Message: fmt.Sprintf("unable to load model file: %s", filename)}
	}

	return &SvmModel{object: mdl}, nil
}

// FreeModel will free the underlying svm_model structure
func FreeModel(mdl *SvmModel) error {

	if mdl == nil {
		return SvmError{Message: "nil model when attempting to free an svm model"}
	}

	if mdl.object == nil {
		return SvmError{Message: "model object's internal svm_model pointer is nil when attempting to free an svm model"}
	}

	C.model_free(mdl.object)
	return nil
}

// FreeParam will free the underlying svm_parameter structure
func FreeParam(param *SvmParameter) error {

	if param == nil {
		return &SvmError{Message: "nil param when attempting to free an svm parameter"}
	}

	if param.object == nil {
		return &SvmError{Message: "param object's internal svm_parameter pointer is nil when attempting to free an svm parameter"}
	}

	C.svm_destroy_param(param.object)
	return nil
}

// Save the model to disk.
// This will return a generic error message if it is unable to save to disk
func (mdl *SvmModel) Save(filename string) error {
	if mdl == nil {
		return SvmError{Message: "nil model when attempting to save an svm model"}
	}

	if mdl.object == nil {
		return SvmError{Message: "model object's internal svm_model pointer is nil when attempting to save an svm model"}
	}

	cfn := C.CString(filename)
	defer C.free(unsafe.Pointer(cfn))

	cerr := C.svm_save_model(cfn, mdl.object)
	if cerr != 0 {
		return SvmError{Message: fmt.Sprintf("unable to save model to file: %s", filename)}
	}

	return nil
}

// Error will return the error message for the error object
func (err SvmError) Error() string {
	return err.Message
}

// Predict will use the model to predict the next values based on the inputs in the SvmNode object
func (mdl *SvmModel) Predict(node *SvmNode) (float64, error) {
	if mdl == nil {
		return -1, SvmError{Message: "nil model when attempting to predict using an svm model"}
	}

	if mdl.object == nil {
		return -1, SvmError{Message: "model object's internal svm_model pointer is nil when attempting to predict using an svm model"}
	}

	if node == nil {
		return -1, SvmError{Message: "nil node when attempting to predict using an svm model"}
	}

	if node.object == nil {
		return -1, SvmError{Message: "node object's internal svm_node pointer is nil when attempting to predict using an svm model"}
	}

	return float64(C.svm_predict(mdl.object, node.object)), nil
}

func (mdl *SvmModel) PredictValues() []float64 {
	return nil
}
