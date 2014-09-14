// Simple LIBSVM wrapper for Go.
package libsvm

/*
#cgo LDFLAGS: -lsvm -lm
#include <svm.h>
*/

import (
	"C"
	"fmt"
	"unsafe"
)

type SvmError struct {
	Message string
}

type SvmProblem struct {
	object *C.svm_problem
}

type SvmParameter struct {
	object *C.svm_parameter
}

type SvmModel struct {
	object *C.svm_model
}

type SvmNode struct {
	object *C.svm_node
}

// Version will return the libsvm version
func Version() int {
	return C.lib_svm_version
}

// Train a model for the given problem using the provided parameters.
// Will return a model or an error
func Train(prob SvmProblem, param SvmParameter) (SvmModel, error) {
	mdl := C.svm_train(prob.object, param.object)
	if mdl == nil {
		return nil, &SvmError{Message: "error while training. nil model returned"}
	}

	return SvmModel{object: mdl}, nil
}

// Load a model from disk. This will return an error message if
// there is a problem loading from disk.
func Load(filename string) (SvmModel, error) {

	cfn := C.CString(filename)
	defer C.free(unsafe.Pointer(cfn))

	mdl := C.svm_load_model(cfn)
	if mdl == nil {
		return nil, &SvmError{Message: fmt.Sprintf("unable to load model file: %s", filename)}
	}

	return SvmModel{object: mdl}, nil
}

// FreeModel will free the underlying svm_model structure
func FreeModel(mdl *SvmModel) {

	if mdl == nil {
		return &SvmError{Message: "nil model when attempting to free an svm model"}
	}

	if mdl.object == nil {
		return &SvmError{Message: "model object's internal svm_model pointer is nil when attempting to free an svm model"}
	}

	C.svm_free_model_content(mdl.object)
}

// FreeParam will free the underlying svm_parameter structure
func FreeParam(param *SvmParameter) {

	if param == nil {
		return &SvmError{Message: "nil param when attempting to free an svm parameter"}
	}

	if param.object == nil {
		return &SvmError{Message: "param object's internal svm_parameter pointer is nil when attempting to free an svm parameter"}
	}

	C.svm_destroy_param(param.object)
}

// Save the model to disk.
// This will return a generic error message if it is unable to save to disk
func (mdl *SvmModel) Save(filename string) error {
	if mdl == nil {
		return &SvmError{Message: "nil model when attempting to save an svm model"}
	}

	if mdl.object == nil {
		return &SvmError{Message: "model object's internal svm_model pointer is nil when attempting to save an svm model"}
	}

	cfn := C.CString(filename)
	defer C.free(unsafe.Pointer(cfn))

	cerr := C.svm_save_model(cfn, mdl.object)
	if cerr != 0 {
		return &SvmError{Message: fmt.Sprintf("unable to save model to file: %s", filename)}
	}

	return nil
}

// Error will return the error message for the error object
func (err SvmError) Error() string {
	return err.Message
}

// Predict will use the model to predict the next values based on the inputs in the SvmNode object
func (mdl *SvmModel) Predict(node SvmNode) float64 {
	if mdl == nil {
		return &SvmError{Message: "nil model when attempting to predict using an svm model"}
	}

	if mdl.object == nil {
		return &SvmError{Message: "model object's internal svm_model pointer is nil when attempting to predict using an svm model"}
	}

	if node == nil {
		return &SvmError{Message: "nil node when attempting to predict using an svm model"}
	}

	if node.object == nil {
		return &SvmError{Message: "node object's internal svm_node pointer is nil when attempting to predict using an svm model"}
	}

	return float64(C.svm_predict(mdl.object, node.object))
}

func (mdl *SvmModel) PredictValues() []float64 {
}
