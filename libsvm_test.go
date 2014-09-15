package libsvm

import (
	"fmt"
	"math"
	"testing"
)

func TestTrain(t *testing.T) {
}

func TestSimpleLoad(t *testing.T) {
	mdl, err := Load("testdata/a1a.model")
	if err != nil {
		t.Error("Error was non-nil", err)
	}

	if mdl == nil {
		t.Error("Error the returned model was nil")
	}
}

func TestLoadAndPredict(t *testing.T) {
	mdl, err := Load("testdata/a1a.model")
	if err != nil {
		t.Error("Model load error was non-nil", err)
	}

	if mdl == nil {
		t.Error("Error the reaturned model was nil")
	}

	exa := NewExample(1, []float64{1, 0, 0, 0, 1, 1, 1})
	nv, nerr := mdl.Predict(exa)
	if nerr != nil {
		t.Error("Predict error result was non-nil", nerr)
	}

	if math.IsNaN(nv) || math.IsInf(nv, 0) {
		t.Error(fmt.Sprintf("Predicted value is NaN or Infinity: %d", nv))
	}
}
