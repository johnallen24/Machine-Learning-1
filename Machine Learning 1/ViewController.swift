//
//  ViewController.swift
//  Machine Learning 1
//
//  Created by John Allen on 7/6/18.
//  Copyright © 2018 jallen.studios. All rights reserved.
//

import UIKit
import AVKit
import Vision

class ViewController: UIViewController {
    
    // Live Camera Properties
    let captureSession = AVCaptureSession()
    var captureDevice:AVCaptureDevice!
    var devicePosition: AVCaptureDevice.Position = .back
    
    let previewView: PreviewView = {
       let pv = PreviewView()
        pv.translatesAutoresizingMaskIntoConstraints = false
        return pv
    }()
    
    let objectTextView: UITextView = {
        let tv = UITextView()
        tv.translatesAutoresizingMaskIntoConstraints = false
        return tv
    }()
    
    var requests = [VNRequest]()
    
  
     var previousPixelBuffer: CVPixelBuffer?
    
     var currentlyAnalyzedPixelBuffer: CVPixelBuffer?
    
     let visionQueue = DispatchQueue(label: "phone")


    override func viewDidLoad() {
        super.viewDidLoad()
        
        view.addSubview(objectTextView)
        NSLayoutConstraint.activate([
            objectTextView.topAnchor.constraint(equalTo: view.topAnchor),
            objectTextView.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            objectTextView.trailingAnchor.constraint(equalTo: view.trailingAnchor),
            objectTextView.heightAnchor.constraint(equalTo: view.heightAnchor, multiplier: 0.1)
            ])
        
        view.addSubview(previewView)
        NSLayoutConstraint.activate([
            previewView.topAnchor.constraint(equalTo: objectTextView.topAnchor),
            previewView.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            previewView.trailingAnchor.constraint(equalTo: view.trailingAnchor),
            previewView.bottomAnchor.constraint(equalTo: view.bottomAnchor)
            ])
        
         setupVision()
    }
    
    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
        prepareCamera()
    }
    
    
    func setupVision() {
        
        // Object Classification
        guard let visionModel = try? VNCoreMLModel(for: Inceptionv3().model) else {fatalError("cant load Vision ML model")}
        
        let classificationRequest = VNCoreMLRequest(model: visionModel, completionHandler: handleClassification)
        classificationRequest.imageCropAndScaleOption = .centerCrop
        
        self.requests = [classificationRequest]
    }
    
    func handleClassification (request:VNRequest, error:Error?) {
        guard let observations = request.results else {print("no results:\(String(describing: error?.localizedDescription))"); return}
        
        let classifcations = observations[0...4]
            .flatMap({$0 as? VNClassificationObservation})
            .filter({$0.confidence > 0.3})
            .map({$0.identifier})
        
        for classification in classifcations {
            DispatchQueue.main.async {
                self.objectTextView.text = classification
            }
        }
        
    }
    
    func analyzeCurrentImage() {
        
        var requestOptions:[VNImageOption:Any] = [:]
        
        if let cameraIntrinsicData = CMGetAttachment(currentlyAnalyzedPixelBuffer!, kCMSampleBufferAttachmentKey_CameraIntrinsicMatrix, nil) {
            requestOptions = [.cameraIntrinsics:cameraIntrinsicData]
        }
        
        let exifOrientation = self.exifOrientationFromDeviceOrientation()
        
        let imageRequestHandler = VNImageRequestHandler(cvPixelBuffer: currentlyAnalyzedPixelBuffer!, orientation: CGImagePropertyOrientation(rawValue:UInt32(exifOrientation))!, options: requestOptions)
        visionQueue.async {
            do {
                defer {self.currentlyAnalyzedPixelBuffer = nil}
                try imageRequestHandler.perform(self.requests)
            } catch {
                print("Error: Vision Request failed with error \(error)")
            }
        }
       
    }
    
    func highlightLogo(boundingRect: CGRect) {
        let source = self.previewView.frame
        
        let rectWidth = source.size.width * boundingRect.size.width
        let rectHeight = source.size.height * boundingRect.size.height
        
        let outline = CALayer()
        outline.frame = CGRect(x: boundingRect.origin.x * source.size.width, y:boundingRect.origin.y * source.size.height, width: rectWidth, height: rectHeight)
        
        
        outline.borderWidth = 2.0
        outline.borderColor = UIColor.red.cgColor
        
        self.previewView.videoPreviewLayer.addSublayer(outline)
    }
    
    
     func predictionsFromMultiDimensionalArrays(observations: [VNCoreMLFeatureValueObservation]?, nmsThreshold: Float = 0.5) -> [Prediction]? {
        guard let results = observations else {
            return nil
        }
        
        let coordinates = results[0].featureValue.multiArrayValue!
        let confidence = results[1].featureValue.multiArrayValue!
        
        let confidenceThreshold = 0.25
        var unorderedPredictions = [Prediction]()
        let numBoundingBoxes = confidence.shape[0].intValue
        let numClasses = confidence.shape[1].intValue
        let confidencePointer = UnsafeMutablePointer<Double>(OpaquePointer(confidence.dataPointer))
        let coordinatesPointer = UnsafeMutablePointer<Double>(OpaquePointer(coordinates.dataPointer))
        for b in 0..<numBoundingBoxes {
            var maxConfidence = 0.0
            var maxIndex = 0
            for c in 0..<numClasses {
                let conf = confidencePointer[b * numClasses + c]
                if conf > maxConfidence {
                    maxConfidence = conf
                    maxIndex = c
                }
            }
            if maxConfidence > confidenceThreshold {
                let x = coordinatesPointer[b * 4]
                let y = coordinatesPointer[b * 4 + 1]
                let w = coordinatesPointer[b * 4 + 2]
                let h = coordinatesPointer[b * 4 + 3]
                
                let rect = CGRect(x: CGFloat(x - w/2), y: CGFloat(y - h/2),
                                  width: CGFloat(w), height: CGFloat(h))
                
                let prediction = Prediction(labelIndex: maxIndex,
                                            confidence: Float(maxConfidence),
                                            boundingBox: rect)
                unorderedPredictions.append(prediction)
            }
        }
        
        var predictions: [Prediction] = []
        let orderedPredictions = unorderedPredictions.sorted { $0.confidence > $1.confidence }
        var keep = [Bool](repeating: true, count: orderedPredictions.count)
        for i in 0..<orderedPredictions.count {
            if keep[i] {
                predictions.append(orderedPredictions[i])
                let bbox1 = orderedPredictions[i].boundingBox
                for j in (i+1)..<orderedPredictions.count {
                    if keep[j] {
                        let bbox2 = orderedPredictions[j].boundingBox
                        if  IoU(bbox1, bbox2) > nmsThreshold {
                            keep[j] = false
                        }
                    }
                }
            }
        }
        
        return predictions
    }

     public func IoU(_ a: CGRect, _ b: CGRect) -> Float {
        let intersection = a.intersection(b)
        let union = a.union(b)
        return Float((intersection.width * intersection.height) / (union.width * union.height))
    }

}

