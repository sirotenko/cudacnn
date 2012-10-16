function varargout = norb_tool(varargin)
% A GUI-demo for general object recogniton example.
%
% Demo needed the NORB dataset
%
% Last update: 2012-04-09


mInputArgs      =   varargin;   % Command line arguments when invoking the GUI
mOutputArgs     =   {};         % Variable for storing output when GUI returns
mIconCData      =   [];         % The icon CData edited by this GUI of dimension
                                % [mIconHeight, mIconWidth, 3]
                                % ~ rot90(im2sim,3)
label = 1;                      % current label of test image (real label)
im2sim          =   [];         % image for testing recognition (to run CNN)

mIsEditingIcon  =   false;      % Flag for indicating whether the current mouse 
                                % move is used for editing color or not
% Variables for supporting custom property/value pairs
mPropertyDefs   =   {...        % The supported custom property/value pairs of this GUI
                     'iconwidth',   @localValidateInput, 'mIconWidth';
                     'iconheight',  @localValidateInput, 'mIconHeight';
                     'TestingImgFile',   @localValidateInput, 'mTestingImgFile'};
mIconWidth      =   100;         % Use input property 'iconwidth' to initialize
mIconHeight     =   100;         % Use input property 'iconheight' to initialize
mTestingImgFile      =   './NORB/norb-5x01235x9x18x6x2x108x108-testing-01-dat.mat'; %fullfile(matlabroot,'./'); 
Images = {0};                    % testing images
Labels = [];                     % images real labels 
im_ptr = 1;
autorecognition = true;

numberOfSamples = 500; % how much sampales load from norb file

cnnet_struct = load('norb_trained_cnn.mat');
cnnet = cnn(cnnet_struct.cnet, false);

norb_objects = load('norb_objects.mat');
norb_objects = norb_objects.norb_objects;

% Create all the UI objects in this GUI here so that they can
% be used in any functions in this GUI
hMainFigure     =   figure(...
                    'Units','pixels',...
                    'MenuBar','none',...
                    'Toolbar','none',...
                    'Position',[300 200 800 600],...
                    'WindowStyle', 'normal',...
                    'WindowButtonDownFcn', @hMainFigureWindowButtonDownFcn,...
                    'WindowButtonUpFcn', @hMainFigureWindowButtonUpFcn,...
                    'WindowButtonMotionFcn', @hMainFigureWindowButtonMotionFcn);
hIconEditPanel  =    uipanel(...
                    'Parent',hMainFigure,...
                    'Units','pixels',...
                    'Clipping','on',...
                    'Position',[50 70 450 460]);
hIconEditAxes   =   axes(...
                    'Parent',hIconEditPanel,...
                    'vis','off',...
                    'Units','pixels',...
                    'Position',[25 25 400 400]);
hIconFileText   =   uicontrol(...
                    'Parent',hMainFigure,...
                    'Units','pixels',...
                    'HorizontalAlignment','left',...
                    'Position',[30 545 100 30],...
                    'FontSize', 10, ...
                    'String','NORB .dat file: ',...
                    'Style','text');
hIconFileEdit   =   uicontrol(...
                    'Parent',hMainFigure,...
                    'Units','pixels',...
                    'HorizontalAlignment','left',...
                    'Position',[130 550 570 30],...
                    'String','Create a new icon or type in an icon image file for editing',...
                    'Enable','inactive',...
                    'Style','edit',...
                    'ButtondownFcn',@hIconFileEditButtondownFcn,...
                    'Callback',@hIconFileEditCallback);
hIconFileButton =   uicontrol(...
                    'Parent',hMainFigure,...
                    'Units','pixels',...
                    'Callback',@hIconFileButtonCallback,...
                    'Position',[710 550 50 30],...
                    'String','...',...
                    'TooltipString','Import From Image File');
hPreviewPanel   =   uipanel(...
                    'Parent',hMainFigure,...
                    'Units','pixels',...
                    'Title','Preview',...
                    'Clipping','on',...
                    'Position',[550 290 200 240]);

hPreviewDesc =   uicontrol(...
                    'Parent',hPreviewPanel,...
                    'Units','pixels',...
                    'Style','text', ...
                    'HorizontalAlignment', 'center',...
                    'Enable','inactive',...
                    'Visible','on',...
                    'FontSize', 14, ...
                    'Position',[10 190 180 30],...
                    'String','');
hPreviewControl =   uicontrol(...
                    'Parent',hPreviewPanel,...
                    'Units','pixels',...
                    'Enable','inactive',...
                    'Visible','off',...
                    'Position',[25 75 100 100],...
                    'String','');
                
hPrevDigitButton =   uicontrol(...
                    'Parent',hPreviewPanel,...
                    'Units','pixels',...
                    'Position',[25 20 50 30],...
                    'String','<',...
                    'Callback',@hPrevDigitButtonCallback);

hNextDigitButton =   uicontrol(...
                    'Parent',hPreviewPanel,...
                    'Units','pixels',...
                    'Position',[125 20 50 30],...
                    'String','>',...
                    'Callback',@hNextDigitButtonCallback);

hAutorecognitionCheck = uicontrol(...
                    'Parent', hMainFigure,...
                    'Units', 'pixels',...
                    'HorizontalAlignment', 'left',...
                    'Position',[550 250 200 30],...
                    'String','Autorecoginition',...
                    'Style','CheckBox', ...
                    'Callback',@hAutorecognitionCheckCallback, ...
                    'Value', autorecognition);
                
hResultPanel   =    uipanel(...
                    'Parent',hMainFigure,...
                    'Units','pixels',...
                    'Title','Recognition result',...
                    'Clipping','on',...
                    'Position',[550 70 200 170]);
                

                
hResultDesc =   uicontrol(...
                    'Parent',hResultPanel,...
                    'Units','pixels',...
                    'Style','text', ...
                    'HorizontalAlignment', 'center',...
                    'Enable','inactive',...
                    'Visible','on',...
                    'FontSize', 15, ...
                    'Position',[10 115 180 30],...
                    'String','');
                
hResultIcon =   uicontrol(...
                    'Parent',hResultPanel,...
                    'Units','pixels',...
                    'Enable','inactive',...
                    'Visible','off',...
                    'Position',[75 35 50 50],...
                    'String','');                
                
for i = 1:numel(norb_objects),
    hResultAlphas(i) = uicontrol(...
        'Parent',hResultPanel,...
        'Units','pixels',...
        'Enable','inactive',...
        'Style','text', ...
        'FontSize', 14, ...
        'Position',[20+(i-1)*160.0/(numel(norb_objects)) 15 20 25],...
        'String', norb_objects{i}.desc(1), ...
        'ForegroundColor', [1 1 1],...
        'BackgroundColor', [1 1 1]);
end

hOKButton       =   uicontrol(...
                    'Parent',hMainFigure,...
                    'Units','pixels',...
                    'Position',[550 15 100 40],...
                    'String','OK',...
                    'Callback',@hOKButtonCallback);
hCancelButton   =   uicontrol(...
                    'Parent',hMainFigure,...
                    'Units','pixels',...
                    'Position',[660 15 100 40],...
                    'String','Cancel',...
                    'Callback',@hCancelButtonCallback);
hRecognizeButton   =   uicontrol(...
                    'Parent',hMainFigure,...
                    'Units','pixels',...
                    'Position',[150 15 100 40],...
                    'String','Recognize',...
                    'Callback',@hRecognizeButtonCallback);
hClearButton   =   uicontrol(...
                    'Parent',hMainFigure,...
                    'Units','pixels',...
                    'Position',[300 15 100 40],...
                    'String','Clear',...
                    'Callback',@hClearButtonCallback);
                
% Host the ColorPalette in the PaletteContainer and keep the function
% handle for getting its selected color for editing icon


% Make changes needed for proper look and feel and running on different
% platforms 
prepareLayout(hMainFigure);                            

% Process the command line input arguments supplied when the GUI is
% invoked 
processUserInputs();                            

% Initialize the iconEditor using the defaults or custom data given through
% property/value pairs
localUpdateIconPlot();

if autorecognition,
    hRecognizeButtonCallback();
end

% Make the GUI on screen
set(hMainFigure,'visible', 'on');
movegui(hMainFigure,'onscreen');

% Make the GUI blocking
uiwait(hMainFigure);

% Return the edited icon CData if it is requested
mOutputArgs{1} =mIconCData;
if nargout>0
    [varargout{1:nargout}] = mOutputArgs{:};
end

    %------------------------------------------------------------------
    function hMainFigureWindowButtonDownFcn(hObject, eventdata)
    % Callback called when mouse is pressed on the figure. Used to change
    % the color of the specific icon data point under the mouse to that of
    % the currently selected color of the colorPalette
        if (ancestor(gco,'axes') == hIconEditAxes)
            mIsEditingIcon = true;
            localEditColor();
        end
    end

    %------------------------------------------------------------------
    function hMainFigureWindowButtonUpFcn(hObject, eventdata)
    % Callback called when mouse is release to exit the icon editing mode
        mIsEditingIcon = false;
    end

    %------------------------------------------------------------------
    function hMainFigureWindowButtonMotionFcn(hObject, eventdata)
    % Callback called when mouse is moving so that icon color data can be
    % updated in the editing mode
        if (ancestor(gco,'axes') == hIconEditAxes)
            localEditColor();
        end
    end

    %------------------------------------------------------------------
    function hIconFileEditCallback(hObject, eventdata)
    % Callback called when user has changed the icon file name from which
    % the icon can be loaded
        file = get(hObject,'String');
        if exist(file, 'file') ~= 2
            errordlg(['The given icon file cannot be found ' 10, file], ...
                    'Invalid Icon File', 'modal');
            set(hObject, 'String', mTestingImgFile);
        else
            mIconCData = [];
            im2sim = [];
            localUpdateIconPlot(); 
            
            if autorecognition,
                hRecognizeButtonCallback();
            end
        end
    end

    %------------------------------------------------------------------
    function hIconFileEditButtondownFcn(hObject, eventdata)
    % Callback called the first time the user pressed mouse on the icon
    % file editbox 
        set(hObject,'String','');
        set(hObject,'Enable','on');
        set(hObject,'ButtonDownFcn',[]);        
        uicontrol(hObject);
    end

    %------------------------------------------------------------------
    function hOKButtonCallback(hObject, eventdata)
    % Callback called when the OK button is pressed
        uiresume;
        delete(hMainFigure);
    end

    %------------------------------------------------------------------
    function hCancelButtonCallback(hObject, eventdata)
    % Callback called when the Cancel button is pressed
        mIconCData =[];
        im2sim = [];
        uiresume;
        delete(hMainFigure);
    end
    %------------------------------------------------------------------
    function hRecognizeButtonCallback(hObject, eventdata)
        if (~isempty(im2sim))
            out = cudacnnMex(cnnet,'sim',single(im2sim));
            max_out = max(out);

            object_id = find(out == max_out, 1); 

            if (object_id <= size(norb_objects,2))

                object_desc = norb_objects{object_id}.desc;
                object_img = norb_objects{object_id}.icon;
                % update preview control
                rows = size(object_img, 1);
                cols = size(object_img, 2);           
                previewSize = getpixelposition(hResultPanel);
                % compensate for the title
                c_cut = 0;
                previewSize(4) = previewSize(4) - c_cut;
                controlWidth = previewSize(3);
                controlHeight = previewSize(4);  
                controlMargin = 15;
                if rows+controlMargin<controlHeight
                    controlHeight = rows+controlMargin;
                end
                if cols+controlMargin<controlWidth
                    controlWidth = cols+controlMargin;
                end

                setpixelposition(hResultIcon,[(previewSize(3)-controlWidth)/2, c_cut + (previewSize(4)-controlHeight)/2, controlWidth, controlHeight]); 
                set(hResultIcon,'CData', object_img,'Visible','on');
                set(hResultDesc, 'string', object_desc);
                
                if (object_id == label)
                    set(hResultDesc, 'ForegroundColor', [0 0.6 0]);
                else
                    set(hResultDesc, 'ForegroundColor', [0.7 0 0])
                end
            else
                set(hResultIcon,'Visible', 'off');
            end
            
            for out_i = 1:numel(out),
                set(hResultAlphas(out_i), 'BackgroundColor', [1 1 1]*(1.8-out(out_i))/3.6);
            end
        else
            disp('nothing to sim');
        end
    end
    %------------------------------------------------------------------
    function hClearButtonCallback(hObject, eventdata)
        mIconCData = ones(mIconHeight, mIconWidth, 3);
        im2sim = ones(mIconWidth, mIconHeight); % ~ rot90(mIconCData)
        localUpdateIconPlot();
        
        if autorecognition,
            hRecognizeButtonCallback(hObject, eventdata)
        end
    end
    %------------------------------------------------------------------
    function hAutorecognitionCheckCallback(hObject, eventdata)
        autorecognition = get(hAutorecognitionCheck, 'Value');
        if autorecognition,
            hRecognizeButtonCallback(hObject, eventdata)
        end

    end
    %------------------------------------------------------------------
    function hIconFileButtonCallback(hObject, eventdata)
    % Callback called when the icon file selection button is pressed
        filespec = {'*.*', 'Database file '};
        [filename, pathname] = uigetfile(filespec, 'Pick an database file', mTestingImgFile);

        if ~isequal(filename,0)
            mTestingImgFile = fullfile(pathname, filename);             
            set(hIconFileEdit, 'ButtonDownFcn',[]);            
            set(hIconFileEdit, 'Enable','on');            
            
            mIconCData = [];
            im2sim = [];
            localUpdateIconPlot();
            
            if autorecognition,
                hRecognizeButtonCallback();
            end
            
        elseif isempty(mIconCData)
            set(hPreviewControl,'Visible', 'off');
            set(hResultIcon,'Visible', 'off');
        end
    end

    function hPrevDigitButtonCallback(hObject, eventdata)
        if(im_ptr>1)
            im_ptr=im_ptr-1;
        end
        label = Labels(im_ptr);
        im2sim = Images{im_ptr};
        im2display = rot90((im2sim - min(im2sim(:)))/(max(im2sim(:)) - min(im2sim(:))), 3);
        mIconCData = cat(3,im2display,im2display,im2display);  
        localUpdateIconPlot();
        
        if autorecognition,
            hRecognizeButtonCallback(hObject, eventdata)
        end        
    end

    function hNextDigitButtonCallback(hObject, eventdata)
        if(im_ptr<numel(Images))
            im_ptr=im_ptr+1;
        end
        label = Labels(im_ptr);
        im2sim = Images{im_ptr};
        im2display = rot90((im2sim - min(im2sim(:)))/(max(im2sim(:)) - min(im2sim(:))), 3);
        mIconCData = cat(3,im2display,im2display,im2display);    
        localUpdateIconPlot();
        
        if autorecognition,
            hRecognizeButtonCallback(hObject, eventdata)
        end
    end
    %------------------------------------------------------------------
    function localEditColor
    % helper function that changes the color of an icon data point to
    % that of the currently selected color in colorPalette 
        if mIsEditingIcon
            pt = get(hIconEditAxes,'currentpoint');

            x = max(1, min(ceil(pt(1,1)), mIconWidth));
            y = max(1, min(ceil(pt(1,2)), mIconHeight));

            % update color of the selected block
            m = get(gcf,'SelectionType');
            r = 3;
            if m(1) == 'n', % left button pressed
                mIconCData(max(y-r,1):min(y+r,end), max(x-r,1):min(x+r,end),:)=0;
                %mIconCData(y, x,:) = 0;
                %if y<mIconHeight,   mIconCData(y+1,x,:) = .8*mIconCData(y+1,x,:); end
                %if x<mIconWidth,    mIconCData(y,x+1,:) = .8*mIconCData(y,x+1,:); end
                %if y>1,             mIconCData(y-1,x,:) = .8*mIconCData(y-1,x,:); end
                %if x>1,             mIconCData(y,x-1,:) = .8*mIconCData(y,x-1,:); end
            else
                mIconCData(max(y-r,1):min(y+r,end), max(x-r,1):min(x+r,end),:)=1;
                %mIconCData(y, x,:) = 1;
            end
            im2sim = rot90(mIconCData(:,:,1));
            localUpdateIconPlot();
        end
    end

    %------------------------------------------------------------------
    function localUpdateIconPlot   
    % helper function that updates the iconEditor when the icon data
    % changes
        %initialize icon CData if it is not initialized
        if isempty(mIconCData)
            if exist(mTestingImgFile, 'file') == 2
                try
                    [Images, Labels] = readNORBImages(mTestingImgFile,numberOfSamples);
                    %im_ptr = 1;
                    label = Labels(im_ptr);
                    im2sim = Images{im_ptr};
                    im2display = rot90((im2sim - min(im2sim(:)))/(max(im2sim(:)) - min(im2sim(:))),3);
                    mIconCData = cat(3,im2display,im2display,im2display);
                    set(hIconFileEdit, 'String', mTestingImgFile);            
                catch
                    errordlg(['Could not load NORB database file successfully. ',...
                              'Make sure the file name is correct: ' mTestingImgFile],...
                              'Invalid NORB File', 'modal');
                    im2sim = [];
                    mIconCData = ones(mIconHeight, mIconWidth, 3);
                end
            else 
                im2sim = [];
                mIconCData = ones(mIconHeight, mIconWidth, 3);
            end
        end
        
        % update preview control
        rows = size(mIconCData, 1);
        cols = size(mIconCData, 2);
        mIconHeight = rows;
        mIconWidth = cols;
        previewSize = getpixelposition(hPreviewPanel);
        % compensate for the title
        
        c_cut = 10;
        
        previewSize(4) = previewSize(4) - c_cut;
        controlWidth = previewSize(3);
        controlHeight = previewSize(4);  
        controlMargin = 15;
        if rows+controlMargin<controlHeight
            controlHeight = rows+controlMargin;
        end
        if cols+controlMargin<controlWidth
            controlWidth = cols+controlMargin;
        end        
        setpixelposition(hPreviewControl,[(previewSize(3)-controlWidth)/2, c_cut + (previewSize(4)-controlHeight)/2, controlWidth, controlHeight]); 
        set(hPreviewControl,'CData', mIconCData,'Visible','on');
        
        % update icon edit pane
        set(hIconEditPanel, 'Title',['Icon Edit Pane (', num2str(rows),' X ', num2str(cols),')']);
        
        s = findobj(hIconEditPanel,'type','surface');        
        if isempty(s)
            gridColor = get(0, 'defaultuicontrolbackgroundcolor') - 0.2;
            gridColor(gridColor<0)=0;
            s=surface('edgecolor','none','parent',hIconEditAxes);
        end        
        %set xdata, ydata, zdata in case the rows and/or cols change
        set(s,'xdata',0:cols,'ydata',0:rows,'zdata',zeros(rows+1,cols+1),'cdata',localGetIconCDataWithNaNs());

        set(hIconEditAxes,'drawmode','fast','xlim',[-.5 cols+.5],'ylim',[-.5 rows+.5]);
        axis(hIconEditAxes, 'ij', 'off');  
        
        preview_desc = norb_objects{label}.desc;
        set(hPreviewDesc, 'string', preview_desc);
        
    end

    %------------------------------------------------------------------
	function cdwithnan = localGetIconCDataWithNaNs()
		% Add NaN to edge of mIconCData so the entire icon renders in the
		% drawing pane.  This is necessary because of surface behavior.
		cdwithnan = mIconCData;
		cdwithnan(:,end+1,:) = NaN;
		cdwithnan(end+1,:,:) = NaN;
		
	end

    %------------------------------------------------------------------
    function processUserInputs
    % helper function that processes the input property/value pairs 
        % Apply possible figure and recognizable custom property/value pairs
        for index=1:2:length(mInputArgs)
            if length(mInputArgs) < index+1
                break;
            end
            match = find(ismember({mPropertyDefs{:,1}},mInputArgs{index}));
            if ~isempty(match)  
               % Validate input and assign it to a variable if given
               if ~isempty(mPropertyDefs{match,3}) && mPropertyDefs{match,2}(mPropertyDefs{match,1}, mInputArgs{index+1})
                   assignin('caller', mPropertyDefs{match,3}, mInputArgs{index+1}) 
               end
            else
                try 
                    set(topContainer, mInputArgs{index}, mInputArgs{index+1});
                catch
                    % If this is not a valid figure property value pair, keep
                    % the pair and go to the next pair
                    continue;
                end
            end
        end        
    end

    %------------------------------------------------------------------
    function isValid = localValidateInput(property, value)
    % helper function that validates the user provided input property/value
    % pairs. You can choose to show warnings or errors here.
        isValid = false;
        switch lower(property)
            case {'iconwidth', 'iconheight'}
                if isnumeric(value) && value >0
                    isValid = true;
                end
            case 'TestingImgFile'
                if exist(value,'file')==2
                    isValid = true;                    
                end
        end
    end
end % end of iconEditor

%------------------------------------------------------------------
function prepareLayout(topContainer)
% This is a utility function that takes care of issues related to
% look&feel and running across multiple platforms. You can reuse
% this function in other GUIs or modify it to fit your needs.
    allObjects = findall(topContainer);
    warning off  %Temporary presentation fix
    try
        titles=get(allObjects(isprop(allObjects,'TitleHandle')), 'TitleHandle');
        allObjects(ismember(allObjects,[titles{:}])) = [];
    catch
    end
    warning on

    % Use the name of this GUI file as the title of the figure
    defaultColor = get(0, 'defaultuicontrolbackgroundcolor');
    if isa(handle(topContainer),'figure')
        set(topContainer,'Name', mfilename, 'NumberTitle','off');
        % Make figure color matches that of GUI objects
        set(topContainer, 'Color',defaultColor);
    end

    % Make GUI objects available to callbacks so that they cannot
    % be changes accidentally by other MATLAB commands
    set(allObjects(isprop(allObjects,'HandleVisibility')), 'HandleVisibility', 'Callback');

    % Make the GUI run properly across multiple platforms by using
    % the proper units
    if strcmpi(get(topContainer, 'Resize'),'on')
        set(allObjects(isprop(allObjects,'Units')),'Units','Normalized');
    else
        set(allObjects(isprop(allObjects,'Units')),'Units','Characters');
    end

    % You may want to change the default color of editbox,
    % popupmenu, and listbox to white on Windows 
    if ispc
        candidates = [findobj(allObjects, 'Style','Popupmenu'),...
                           findobj(allObjects, 'Style','Edit'),...
                           findobj(allObjects, 'Style','Listbox')];
        set(findobj(candidates,'BackgroundColor', defaultColor), 'BackgroundColor','white');
    end
end

function [I, lbl] = readNORBImages(filepath,num)
    
    norb_test_reader.num_samples = num;
    norb_test_reader.current = 1;
    norb_test_reader.data_files = {filepath};
    norb_test_reader.label_files = {strrep(filepath, 'dat.mat', 'cat.mat')};
    norb_test_reader.buffer_size = 1000;
    norb_test_reader.read = @norb_datareader;
    norb_test_reader.stereo = false;

    for k=1:norb_test_reader.num_samples
        [inp, targ, norb_test_reader] = norb_test_reader.read(norb_test_reader);
        I{k} = inp;
        lbl(k) = find(targ == max(targ),1);
    end
    
end
