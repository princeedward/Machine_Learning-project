profile on
% businessID = [train_metadata.business_id];
% uniqueIDs = unique(businessID);
% NumID = zeros(size(businessID,2),1);
% % NumID(businessID == uniqueIDs(1))=1;
% for i = 1:length(uniqueIDs)
%     NumID(strcmp(businessID,uniqueIDs(i))) = i;
% end
% userID = [train_metadata.user_id];
% uniqueUserIDs = unique(userID);
% UserNumID = zeros(size(userID,2),1);
% % NumID(businessID == uniqueIDs(1))=1;
% for i = 1:length(uniqueUserIDs)
%     UserNumID(strcmp(userID,uniqueUserIDs(i))) = i;
% end
% Predictlabel = zeros(size(train.labels));
% for i = 1:length(uniqueIDs)
%     ResRate(i) = mean(train.labels(NumID==i));
%     Predictlabel(NumID==i) = round(ResRate(i));
% end
% quizBusinessID = [quiz_metadata.business_id];
% QuizNumID = zeros(size(quizBusinessID,2),1);
% for i = 1:length(uniqueIDs)
%     QuizNumID(strcmp(quizBusinessID,uniqueIDs(i))) = i;
% end
% rates = zeros(size(quiz.counts,1),1);
% for i = 1:length(uniqueIDs)
% %     ResRate(i) = mean(train.labels(NumID==i));
%     rates(QuizNumID==i) = round(ResRate(i));
% end

% quizUserID = [quiz_metadata.user_id];
% QuizUserNumID = zeros(size(quizUserID,2),1);
% for i = 1:length(uniqueUserIDs)
%     QuizUserNumID(strcmp(quizUserID,uniqueUserIDs(i))) = i;
% end
% NewRates = zeros(sum(rates==0),1);
% ObIDX = find(rates==0);
% for i = 1:length(ObIDX)
%     NewRates(i) = round(mean(train.labels(UserNumID==QuizUserNumID(ObIDX(i)))));
% end
% rates(rates==0) = NewRates;
rates(isnan(rates)) = 4;
profile off
profile viewer