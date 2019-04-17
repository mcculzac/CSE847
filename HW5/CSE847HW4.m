function CSE847HW4()

    function err = error(X, c, idx)
        err = 0;
        for n=1:size(X,0)
            err = err + norm(c(idx(n), :) - X(n), :)^2; % squared sum error
        end
    end

    function err = difference(old_c, c)
        err = 0;
        for n=1:size(c)
            err = err + abs(norm(c(n) - old_c(n)));
        end
    end

    function c = update_centers(X, idx)
        % disp(size(X, 1));
        idxmax = max(idx);
        x2size = size(X, 2);
        c = zeros(max(idx),size(X, 2));  % max of idx is # of clusters
        disp('C!');
        disp(c);
        counts = zeros(idxmax, 1);
        xsizey = size(X);
        idxsize = size(idx);
        % csize = size(c);
        for n=1:size(X, 1)
            % disp('c size');
            % disp(size(c));
            % disp('n');
            % disp(n)
            % disp('idx of n');
            % disp(idx(n));
            disp('thing im adding!');
            disp(c(idx(n), :) + X(n));
            c(idx(n), :) = c(idx(n), :) + X(n, :); % add x data point to 
            counts(idx(n)) = counts(idx(n)) + 1; % add membership
        end
        
        % disp('counts');
        % disp(counts)
        % disp('cpre');
        % disp(c);
        % divide
        for n=1:size(counts)
            % disp('idx(n)');
            
            % disp('counts(idx(n))');
            % disp(counts(idx(n)));
            c(n, :) = c(n, :)./counts(n);
        end
        disp('cpost');
        disp(c);
    end
    
    function idx = update_memberships(X, c)
        idx = zeros(size(X, 1), 1);
        
        for i=1:size(X, 1)
            temp = zeros(size(c, 1), 1);
            for j=1:size(temp, 1)
                temp(j) = norm(X(i, :)-c(j, :))^2;
            end
            [~, min_indx] = min(temp);  % only care about min index
            idx(i) = min_indx;
            % disp('temp/idx');
            % disp(temp);
            % disp(idx);
        end
    end

    function [idx, c] = cust_kmeans(X, k, tol)
        c = [];
        idx = [];
        % disp('X');
        % disp(X);
        % initiate first center
        c = [X(randi(size(X, 1)), :)];
        for n=1:k-1
            temp = zeros(size(X,1),1);
            for l=1:size(X, 1)
                c_dists = zeros(size(c,1), 1);
                for m=1:size(c,1)
                    c_dists(m) = (norm(X(l)-c(m)))^2;
                end
                % disp('m: ');
                % disp(m);
                % disp('cdists');
                % disp(c_dists);
                % c dists now has distance from point to all centroids
                [~, temp_c_min_idx] = min(c_dists);
                closest = c(temp_c_min_idx);
                % disp('closest centroid');
                % disp(closest);
                % temp will get distance from closest centroiid for point l
                temp(l) = norm(X(l)-closest)^2;
            end
            % disp('temps');
            % disp(temp);
            [~, max_dist_idx] = max(temp);
            % disp('max dist idx');
            % disp(max_dist_idx);
            % disp('X');
            % disp(X);
            % disp('c');
            % disp(c);
            c = [c; X(max_dist_idx, :)];
            
        end
        % disp('initalized c:');
        %  disp(c);
        idx = update_memberships(X, c);
        dif = tol*10;
        while dif > tol
            old_c = c;
            % disp('cs:');
            % disp(c);
            % disp('idx');
            % disp(idx);
            c = update_centers(X, idx);
            dif = difference(old_c, c);
            % plot_stuff(X, c, idx);
            % disp('dif');
            % disp(dif);
        end
        %  disp('cend:');
        % disp(c);
    end

    function [idx, c, ystar] = spectral_kmeans(X, k, tol, num_eigs)
        x_square = X*X.';
        [reigvs, eigs] = eig(x_square);
        disp('size x');
        disp(size(X));
        % disp('reigvs');
        % disp(reigvs);
        disp('size xsquare');
        disp(size(x_square));
        [eigs,ind] = sort(diag(eigs));
        reigvs = reigvs(:,ind);

        
        ystar = reigvs(:, 1:num_eigs);
        disp('ystar size');
        disp(size(ystar));
        [idx, c] = cust_kmeans(ystar, k, tol);
    end

    sample_data = randn(1000, 3);
    [idx, c] = cust_kmeans(sample_data, 5, .001);
    disp('regular k-means centers');
    disp(c);
    % X = sample_data;
    % threed_plot_stuff(X, c, idx);
    
    
    [idx, c, xs] = spectral_kmeans(sample_data, 3, .001, 2);
    disp('spectral k-means centers in 2d');
    disp(c);
    twod_plot_stuff(xs, c, idx);
    
    function [] = threed_plot_stuff(X, c, idx)
        figure;
        plot3(X(idx==1,1),X(idx==1,2),X(idx==1,3),'r.','MarkerSize',12)
        hold on
        plot3(X(idx==2,1),X(idx==2,2),X(idx==2,3),'b.','MarkerSize',12)
        hold on
        plot3(X(idx==3,1),X(idx==3,2),X(idx==3,3),'g.','MarkerSize',12)
        hold on
        plot3(X(idx==4,1),X(idx==4,2),X(idx==4,3),'c.','MarkerSize',12)
        hold on
        plot3(X(idx==5,1),X(idx==5,2),X(idx==5,3),'m.','MarkerSize',12)
        xlim('auto');
        ylim('auto');
        % plot3(c(:,1),c(:,2),c(:,3),'kx', 'MarkerSize',14,'LineWidth',3) 
        legend('Cluster 1','Cluster 2','Cluster 3', 'Cluster 4', 'Cluster 5', 'Centroids','Location','NE')
        title 'Cluster Assignments and Centroids'
        hold off
    end

    function [] = twod_plot_stuff(X, c, idx)
        figure;
        plot(X(idx==1,1),X(idx==1,2),'r.','MarkerSize',12)
        hold on
        plot(X(idx==2,1),X(idx==2,2),'b.','MarkerSize',12)
        hold on
        plot(X(idx==3,1),X(idx==3,2),'g.','MarkerSize',12)
        hold on
        plot(X(idx==4,1),X(idx==4,2),'c.','MarkerSize',12)
        hold on
        plot(X(idx==5,1),X(idx==5,2),'m.','MarkerSize',12)
        xlim('auto');
        ylim('auto');
        %  plot(c(:,1),c(:,2),'kx','MarkerSize',14,'LineWidth',3) 
        legend('Cluster 1','Cluster 2','Cluster 3', 'Cluster 4', 'Cluster 5', 'Centroids','Location','NE')
        title 'Cluster Assignments and Centroids'
        hold off
    end

end
