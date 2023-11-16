import torch
from temp import FeedForwardNet,download_mnist_data

class_mapping=[
    '0',
    '1',
    '2',
    '3',
    '4',
    '5',
    '6',
    '7',
    '8',
    '9'
    ]


#implement predict function
def predict(model,input,target,class_mapping):
    model.eval()

    with torch.no_grad():
        predictions=model(input)
        predicted_index=predictions[0].argmax(0)
        
       
        predicted=class_mapping[predicted_index]
        
       
        expected=class_mapping[target]
    return predicted,expected


if __name__=="__main__":
    #loadback model
    feed_forward_net=FeedForwardNet()
    
    #load the state dictionary
    state_dict=torch.load('feedforward.pth')
    feed_forward_net.load_state_dict(state_dict)
    
    #load MNIST validation dataset
    _,validation_data=download_mnist_data()
    
    #get a sample from the validation dataset for inference
    input,target=validation_data[0][0],validation_data[0][1]
    
    #make on inference
    predicted,expected=predict(feed_forward_net,
                               input,target,class_mapping
                               )
    print(f'predicted:{predicted},expected:{expected}')