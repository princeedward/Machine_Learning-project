function SVM_result = SVM_classify(X_train,Y_train)
all_labels = unique(Y_train);
number_of_labels = length(all_labels);
[nos_observations,nos_features] = size(X_train);
for a = 1:number_of_labels
    % creating seperate models for each label. 1 is the current label 0 is
    % all others. see
    % http://courses.media.mit.edu/2006fall/mas622j/Projects/aisen-project/
    % one-against-all classification
    current_labels = all_labels(a)*(Y_train==all_labels(a));
    SVM_model_struct(a) = svmtrain(TrainingSet,current_labels);
end

    


