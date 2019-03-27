

function [] = logistic_train(data, labels, epsilon, maxiter)
%
% code to train a logistic regression classifier
%
% INPUTS:
% data = n * (d+1) matrix with n samples and d features, where
%   column d+1 is all ones (corresponding to the intercept term)
% labels = n * 1 vector of class labels (taking values 0 or 1)
% epsilon = optional argument specifying the convergence
% criterion - if the change in the absolute difference in
% predictions, from one iteration to the next, averaged across
% input features, is less than epsilon, then halt
% (if unspecified, use a default value of 1e-5)
% maxiter = optional argument that specifies the maximum number of
% iterations to execute (useful when debugging in case your
% code is not converging correctly!)
% (if unspecified can be set to 1000)
%
% OUTPUT:
% weights = (d+1) * 1 vector of weights where the weights correspond to
% the columns of "data"

    % var def
    if exist('data', 'var') == 0
        data = load('data.txt');
    end
    if exist('labels', 'var') == 0
        labels = load('labels.txt');
    end
    if exist('epsilon', 'var') == 0
        tol = 1e-5;
    else
        tol = epsilon;
    end
    if exist('maxiter', 'var') == 0
        maxiter = 1000;
    end

    % function defs
    
    % gradient descent
    function [w] = graddes(phi, t)
        
        function [dW] = deltaw(w, x, y)
            sum = 0;
            for j=1:length(x)
                sum = sum + (w*x(j)-y(j))*x(j);
            end
            dW = sum;
        end

       
        alpha = .0001;
        iters = 0;
%         t_size = size(t, 1);
          phi_size = size(phi(1));
%         disp(phi_size(1));
          w = zeros(57, 1);
%         disp('w size');
%         disp(size(w));
        avg_dif = 1;
        % disp(strcat('phi ', mat2str(size(phi))));
        % disp(strcat('phi.t', mat2str(size(phi.'))));
        % disp(strcat('t ', mat2str(size(t))));
        % disp(strcat('w ', mat2str(size(w))));
        % disp(phi(1:10, :));
        while and(iters < maxiter, avg_dif > tol)
            
            w0 = w;
            % w = w0 - alpha * deltaw(w0, phi, t);
            % w = w0 - alpha * (w0.'*phi-t)*phi;
            w = w0 - alpha * (phi.'*phi*w0-phi.'*t);
            % disp('w, w0, avg_dif: ');
            % disp(w([1:4]));
            % disp(w0([1:4]));
            avg_dif = abs(mean(w0 - w));
            % disp(avg_dif);
            iters = iters + 1;
        end
        % disp(strcat('w ', mat2str(size(w))));
        % disp(w(1, :))
    end
  
    % make labels 1, -1 and split
    labels(labels<1) = -1;
    t_train = labels(1:2000, :);
    t_test = labels(2001:end, :);
    data_train = data(1:2000, :);
    data_test = data(2001:end, :);
    
    % disp('data_test');
    % disp(size(data_test));
    % disp(strcat('data train: ', mat2str(size(data_train))));
    preds = zeros(6);
    % disp(data_test(1:10, :));
    % disp(t_train(1:10, :));
    
    % run through and test model and print accuracy
    function [] = test_model(w, xtest, ytest)
        right = 0;
        % disp('sizes: ');
        % disp(xtest(1:10));
        % disp(size(w));
        % disp(size(xtest(1)));
        for k=1: length(xtest)
            pred_orig = xtest(k, :) * w;
            % disp(pred_orig);
            if pred_orig > 0
                pred = 1;
            elseif pred_orig < 0
                pred = -1;
            else
                % disp('pred_orig is 0?');
                % disp(pred_orig);
                pred = 0;
            end
            if pred == ytest(k)
                right = right + 1;
            end
        end
        % disp('xtest: ');
        % disp(size(xtest));
        %disp('w: ');
        %disp(w);
        % disp('pred_orig size: ');
        % disp(size(pred_orig));
        % disp('pred: ');
        % disp(pred);
        acc = right / length(xtest);
        disp('accuracy ');
        disp(acc);
    end
    
    % n to use
    ns = [200, 500, 800, 1000, 1500, 2000];
    
    % go through and train and test model
    for i=1: length(ns)
        weights = graddes(data_train(1:ns(i), :), t_train(1:ns(i), :));
        % disp(weights(1, :));
        % disp(strcat('data_test: ', mat2str(size(data_test(1:ns(i), :)))));
        test_model(weights, data_test, t_test);
        
    end
    
    % weights = graddes(data_train, t_train);
    

end